#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import math
import os
import random
import datasets
import torch
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.auto import tqdm
from models.model import GPT
import json
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    DataCollatorForSeq2Seq,
    get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM,
    BitsAndBytesConfig,
)
import time
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

logger = get_logger(__name__)

# 从json文件中读取数据，并放入prompt和completion中，返回的是一个sample{}
def read_data():
    data_list = []

    data_path = '/data1/cchuan/data/mllm/clean_data1.json'

    with open(data_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            data_list.append(data)

    # 现在，data_list 包含了所有的 JSON 对象

    samples = {'prompt':[], 'completion': []}
    for item in data_list[0]['train']:
        samples['prompt'].append(item['input'])
        samples['completion'].append(item['output'])
    
    print('finish reading')

    return samples

# 将上面read_data得到的sample进行tokenize
# 然后返回两个set，是输入经过encode_tokenize和输出经过decode_tokenize之后的结果
def solve_data(samples, max_length):
    encode_tokenizer_path = '/data1/cchuan/data/weight/xlmr/'
    decode_tokenizer_path = '/data1/cchuan/data/weight/tiny_llama'

    encode_tokenizer = AutoTokenizer.from_pretrained(encode_tokenizer_path)
    encode_tokenizer.padding_side = 'right'
    encode_max_length = max_length

    decode_tokenizer = AutoTokenizer.from_pretrained(decode_tokenizer_path)
    decode_tokenizer.pad_token = "$$"
    decode_tokenizer.padding_side = 'right'
    decode_max_length = max_length

    print('finish loading')

    start_time = time.time()

    encoded_data = encode_tokenizer(
        samples['prompt'],
        padding="max_length",
        return_tensors="pt",
        truncation=True,
        max_length=encode_max_length,  # 显式指定最大长度
    )

    time1 = time.time()

    print('finish encode tokenize, cost time {} s'.format(time1 - start_time))

    text = [t + '\n' for t in samples["completion"]]

    decode_data = decode_tokenizer(
        text,
        padding="max_length",
        return_tensors="pt",
        truncation=True,
        max_length=decode_max_length,  # 显式指定最大长度
        add_special_tokens=False,
    )

    time2 = time.time()

    print('finish decode tokenize, cost time {} s'.format(time2 - start_time))

    print('finish tokenize')

    return {'input_ids': encoded_data['input_ids'], 'attention_mask': encoded_data['attention_mask']}, \
        {'input_ids': decode_data['input_ids'], 'attention_mask': decode_data['attention_mask']}


# 这个是自定义的Dataset
class CustomDataset(Dataset):
    def __init__(self, encoded_data, decode_data):
        self.encoded_data = encoded_data
        self.decode_data = decode_data

    def __len__(self):
        return len(self.encoded_data['input_ids'])

    def __getitem__(self, idx):
        encoded_text = \
            {'input_ids': self.encoded_data['input_ids'][idx], 'attention_mask': self.encoded_data['attention_mask'][idx]}
        decode_text = \
            {'input_ids': self.decode_data['input_ids'][idx], 'attention_mask': self.decode_data['attention_mask'][idx]}

        # 这里你可以根据需要进行数据转换、预处理等操作
        # encoded_text_tensor = torch.tensor(encoded_text)
        # decode_text_tensor = torch.tensor(decode_text)

        return {'text': encoded_text, 'answer': decode_text}

# 获得Dataloader
def get_dataloader(batch_size=10, max_length=256, validation_split=0.2):
    # 从文件读入数据
    samples = read_data()
    # 进行tokenize
    encoded_data, decode_data = solve_data(samples, max_length)
    # 变成dataset
    custom_dataset = CustomDataset(encoded_data, decode_data)
    # 转为Dataloader
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

    dataset = data_loader.dataset
    dataset_size = len(dataset)
    validation_size = int(dataset_size * validation_split)
    split_sizes = [dataset_size - validation_size, validation_size]
    train_dataset, val_dataset = random_split(dataset, split_sizes)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader



def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")

    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="If passed, will use LORA (low-rank parameter-efficient training) to train the model.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The rank of lora.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16,
        help="The alpha parameter of lora.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="The dropout rate of lora modules.",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="If passed, will use flash attention to train the model.",
    )

    # 这个参数重要
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=256,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    # batch_size
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    # 学习率
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0, help="Ratio of total training steps used for warmup."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Log the training loss and learning rate every logging_steps steps.",
    )

    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )

    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )

    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=(
            "Turn on gradient checkpointing. Saves memory but slows training."
        ),
    )

    parser.add_argument(
        "--use_qlora",
        action="store_true",
        help=(
            "Use qLoRA training - main thing is initialising model in quantised form. Not compatible with deepspeed."
        ),
    )
    # 梯度裁剪
    parser.add_argument(
        '--clip_grad_norm',
        type=float,
        default=-1,
        help='Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead).',
    )

    parser.add_argument(
        '--use_8bit_optimizer',
        action='store_true',
        help='Use 8bit optimizer from bitsandbytes. Not compatible with deepspeed (use deepspeed config instead).',
    )

    args = parser.parse_args()

    return args


def save_with_accelerate(accelerator, model, output_dir, args):
    unwrapped_model = accelerator.unwrap_model(model)
    # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
    # Otherwise, sometimes the model will be saved with only part of the parameters.
    # Also, accelerator needs to use the wrapped model to get the state_dict.
    state_dict = accelerator.get_state_dict(model)
    if args.use_lora:
        # When using lora, the unwrapped model is a PeftModel, which doesn't support the is_main_process 
        # and has its own save_pretrained function for only saving lora modules.
        # We have to manually specify the is_main_process outside the save_pretrained function.
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained(output_dir, state_dict=state_dict)
    else:
        unwrapped_model.save_pretrained(
            output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=state_dict
        )



# -----------------------------------------------从这里程序开始执行--------------------------------------------------

# 我将原有的模型写成一个较大的模型直接当做model使用
def main():
    # 这里输入各种参数
    # 注意，我们的模型中，含有两个不同的tokenizer和两个不同的model
    # 所以应该自己进行修改！！！
    args = parse_args()

    # A hacky way to make llama work with flash attention
    if args.use_flash_attn:
        from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    # Make one log on every process with the configuration for debugging.

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    accelerator.wait_for_everyone()


# --------------------------------------------上面是各种参数的配置----------------------------------------------------------------------

    # 这里开始正式进行训练!!!

    # 如果模型的路径是ok的话
    # 这里我们先采用不使用LoRA的方法进行fine-tune
    # if args.model_name_or_path:
    if args.use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        device_index = accelerator.local_process_index
        device_map = {"": device_index} # force data-parallel training.
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            load_in_4bit=True,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )
    else:
        model = GPT()


    # 暂时先不考虑使用LoRA
    if args.use_lora:
        if args.use_qlora:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

        logger.info("Initializing LORA model...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=args.lora_rank, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, peft_config)
        # peft breaks flash attention due to casting norms to fp32. This fixes it back up.
        # See https://github.com/huggingface/peft/issues/790
        from llama_flash_attn_monkey_patch import upcast_layer_for_flash_attention
        model = upcast_layer_for_flash_attention(model, torch.bfloat16)
        model.print_trainable_parameters()


# ----------------------------------------开始制作数据集--------------------------------------------------------------------------


    train_dataloader, val_dataloader = \
        get_dataloader(batch_size=args.per_device_train_batch_size, max_length=args.max_seq_length)


# ------------------------------------设置训练的优化器以及学习了参数-------------------------------------------------------------------

    # TO DO (Finish)
    # 设置Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    # 我们先不使用LoRA
    if args.use_qlora:
        from bitsandbytes.optim import AdamW
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            optim_bits=8 if args.use_8bit_optimizer else 32,
            is_paged=True
        )
    # 这里得到了我们需要的优化器
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # 这个不用管
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler for the `num_processes` times. This is because they assume 
    # the user initialize the scheduler with the entire training set. In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set. So each time the process needs to update the lr multiple times so that the total 
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the entire training set (when epochs is specified) or we need to multiply the 
    # num_training_steps by num_processes so that the total number of updates matches the num_training_steps.
    num_training_steps_for_scheduler = args.max_train_steps \
        if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )

    # TO DO (Finish)
    # Prepare everything with `accelerator`.
    print('arrive at accelerator prepare')
    print(model)
    print(optimizer)
    print(train_dataloader)
    print(lr_scheduler)

    # 将模型放到accelerator里面
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler, 
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)


    # 这个是记录存储的东西
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("open_instruct", experiment_config)
        
        
# --------------------------------------开始训练！---------------------------------------------------------------

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    cnt = 0

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                loss = model(batch)
                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                # clip gradient norm. don't do this with deepspeed
                if accelerator.sync_gradients and args.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()       

            # 下面一堆都是几率
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = accelerator.gather(total_loss).mean().item() / args.gradient_accumulation_steps / args.logging_steps
                    logger.info(f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}")
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "train_loss": avg_loss,
                            },
                            step=completed_steps,
                        )
                    total_loss = 0
                    

                if completed_steps >= args.max_train_steps:
                    break

        # 每个epoch记录数据
        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                cnt += 1
                output_path = os.path.join(args.output_dir, str(cnt))
                accelerator.save_model(model, output_path)


if __name__ == "__main__":
    main()
