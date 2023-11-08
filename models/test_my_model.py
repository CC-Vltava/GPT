import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from xlmr import xlmr
from Qformer import BertConfig, BertLMHeadModel
from transformers import AutoTokenizer, AutoModel

import json




# 模型结构
# xlm-r tokenizer
# xlm-r             冻结
# Q-Former          训练
# linear layer      训练
# llama             冻结
# llama tokenizer

GPU_DEVICE='cuda'

class GPT(nn.Module):
    def __init__(
            self,
            num_query_token=32,
            # llama_model_path='/home/cchuan/Project/MiniGPT-4/weight/vicuna/',
            llama_model_path='/data1/cchuan/data/weight/tiny_llama/',
            low_resource=False,  # use 8 bit and put vit in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
            end_sym='\n',
            max_txt_len=128,
            load_model_path='/data1/cchuan/data/weight/model/my_model.pth',
            save_model_path='/data1/cchuan/data/weight/model/my_model.pth',
            print_message=False
        ):
        super(GPT, self).__init__()

        self.print_message = print_message

        self.load_model_path = load_model_path
        self.save_model_path = save_model_path
        self.low_resource = low_resource
        self.device = GPU_DEVICE if torch.cuda.is_available() else "cpu"

        # 加载xlm-r
        if self.print_message:
            print('Loading XLM-R')
        self.xlmr = xlmr()
        # 冻结xlm-r
        self.xlmr.freeze()
        if self.print_message:
            print('Loading XLM-R Done')

        input_features = 1408
        self.xlmr2qformer = nn.Linear(self.xlmr.num_features, input_features)

        # 加载Q-Former
        if self.print_message:
            print('Loading Q-Former')
        # 这里改成xlmr的输出大小
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, input_features
        )
        self.Qformer = self.Qformer
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        # Q-Former不能冻结
        for name, param in self.Qformer.named_parameters():
            param.requires_grad = True
        self.Qformer = self.Qformer.eval()
        self.query_tokens.requires_grad = True
        if self.print_message:
            print('Loading Q-Former Done')


        # 加载LLAMA
        if self.print_message:        
            print('Loading LLAMA')
        # self.llama_model=None
        if self.low_resource:
            self.llama_model = AutoModel.from_pretrained(
                llama_model_path,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = AutoModel.from_pretrained(
                llama_model_path,
                torch_dtype=torch.float16,
            )
        self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_path, use_fast=False)
        self.llama_tokenizer.pad_token = "$$"

        # LLAMA 冻结
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        if self.print_message:
            print('Loading LLAMA Done')


        # 加载线性层
        if self.print_message:
            print('Loading Linear Layer')
        input_size_from_QFormer = self.Qformer.config.hidden_size
        self.llama_proj = nn.Linear(
            input_size_from_QFormer, self.llama_model.config.hidden_size
        )
        if self.print_message:
            print('Loading Linear Layer Done')

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if self.print_message:
            print('Init Done')

    def to_device(self, device):
        torch.cuda.empty_cache()
        self.xlmr = self.xlmr.to(device)
        self.Qformer = self.Qformer.to(device)
        self.llama_model = self.llama_model.to(device)
        self.llama_proj = self.llama_proj.to(device)
        self.xlmr2qformer = self.xlmr2qformer.to(device)
        self.device = device


    def init_Qformer(self, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("/data1/cchuan/data/weight/bert/")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def proceed_in_qformer(self, text):
        # device = 'cpu'
        text_atts = torch.ones(text.size()[:-1], dtype=torch.long).to(self.device)

        query_tokens = self.query_tokens.expand(text.shape[0], -1, -1).to(self.device)
        text = text.to(self.device)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=text,
            encoder_attention_mask=text_atts,
            return_dict=True,
        )
        inputs_llama = self.llama_proj(query_output.last_hidden_state)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device)

        return inputs_llama, atts_llama
    

    # 这个是在做Q-Former训练的时候加入提示
    def prompt_wrap(self, img_embeds, atts_img, prompts):
        if prompts:
            emb_lists = []
            if isinstance(prompts, str):
                prompts = [prompts] * len(img_embeds)

            # 这个for循环，是将prompts内容进行toknize和embedding，然后把embed的图片信息传进去
            for each_img_embed, each_prompt in zip(img_embeds, prompts):
                p_before, p_after = each_prompt.split('<ImageHere>')

                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_before_embed = self.embed_tokens(p_before_tokens.input_ids)
                p_after_embed = self.embed_tokens(p_after_tokens.input_ids)
                wrapped_emb = torch.cat([p_before_embed, each_img_embed[None], p_after_embed], dim=1)
                emb_lists.append(wrapped_emb)
            
            emb_lens = [emb.shape[1] for emb in emb_lists]
            # 创建了一个名为 pad_emb 的嵌入向量，表示填充的特殊嵌入向量
            pad_emb = self.embed_tokens(torch.tensor(self.llama_tokenizer.pad_token_id, device=img_embeds.device))
            wrapped_embs = pad_emb.expand(len(emb_lens), max(emb_lens), -1).clone()
            wrapped_atts = torch.zeros([len(emb_lens), max(emb_lens)], dtype=torch.int, device=img_embeds.device)
            for i, emb in enumerate(emb_lists):
                wrapped_embs[i, :emb_lens[i]] = emb
                wrapped_atts[i, :emb_lens[i]] = 1
            return wrapped_embs, wrapped_atts
        else:
            return img_embeds, atts_img
        

    # 将token进行embedding的过程
    def embed_tokens(self, token_ids):
        # device = 'cpu'
        if hasattr(self.llama_model.base_model, 'model'): ## lora wrapped model
            embeds = self.llama_model.base_model.model.model.embed_tokens(token_ids.to(self.device))
        else:
            embeds = self.llama_model.base_model.embed_tokens(token_ids.to(self.device))
        return embeds

    def concat_emb_input_output(self, input_embs, input_atts, output_embs, output_atts):
        # device = 'cpu'
        input_lens = []
        cat_embs = []
        cat_atts = []
        for i in range(input_embs.size(0)):
            input_len = input_atts[i].sum()
            input_lens.append(input_len)
            cat_embs.append(
                torch.cat([
                    input_embs[i][:input_len].to(self.device),
                    output_embs[i].to(self.device),
                    input_embs[i][input_len:].to(self.device)
                ])
            )
            cat_atts.append(
                torch.cat([
                    input_atts[i][:input_len].to(self.device),
                    output_atts[i].to(self.device),
                    input_atts[i][input_len:].to(self.device)
                ])
            )
        cat_embs = torch.stack(cat_embs)
        cat_atts = torch.stack(cat_atts)
        return cat_embs, cat_atts, input_lens

    # 这里的text是已经过了xlm-r的tokenizer
    def forward(self, samples):
        if self.print_message:
            print('Start GPT')
        text = samples['text']
        # 提取特征
        if self.print_message:
            print('xlm-r')
        a = self.xlmr(text)
        b = a.to(self.device)
        text_features = self.xlmr(text).to(self.device)
        text_features = self.xlmr2qformer(text_features)
        if self.print_message:
            print('Finish xlm-r')

        if self.print_message:
            print('Q-Former')

        img_embeds, atts_img = self.proceed_in_qformer(text_features)

        if self.print_message:
            print('Finish Q-Former')

        if self.print_message:
            print('regress answer')
        # 这个prompt list暂时没有
        # if self.prompt_list:
        #     instruction = random.choice(self.prompt_list)
        # else:
        #     instruction = samples["instruction_input"] if "instruction_input" in samples else None


        # 这个instruction也暂时没有
        # img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, instruction)

        # 在答案右侧加上end_sym（其实就是加上\n)
        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["answer"]]

        # 这个是把答案进行tokenizer
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(self.device)

        # bos是一个整数标识符，表示文本序列的开头
        batch_size = img_embeds.shape[0]
        # 这里为每一个batch创造一个bos
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id

        # 准备好在最前面加入bos
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        # 在这里将tokenize之后的答案进行embedding
        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds, attention_mask, input_lens = \
            self.concat_emb_input_output(img_embeds, atts_img, to_regress_embeds, to_regress_tokens.attention_mask)
        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, attention_mask], dim=1)

        part_targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        targets = (
            torch.ones([inputs_embeds.shape[0], inputs_embeds.shape[1]],
                       dtype=torch.long).to(self.device).fill_(-100)
        )

        for i, target in enumerate(part_targets):
            targets[i, input_lens[i] + 1: input_lens[i] + len(target) + 1] = target  # plus 1 for bos

        if self.print_message:
            print('Finish regress answer')


        inputs_embeds = inputs_embeds.to(self.device).to(torch.float16)
        attention_mask = attention_mask.to(self.device)
        targets = targets.to(self.device)

        if self.print_message:
            print('LLAMA')
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )


        loss_function = nn.CrossEntropyLoss()

        embedding_matrix = self.llama_model.embed_tokens.weight.T
        feature_size = outputs['last_hidden_state'].size(-1)
        my_output = outputs['last_hidden_state'].view(-1, feature_size).to(self.device) @ embedding_matrix
        targets = targets.view(-1).to(self.device)
        
        # 计算损失
        if self.print_message:
            print(my_output.shape, targets.shape)
        loss = loss_function(my_output, targets)

        if self.print_message:
            print("交叉熵损失:", loss.item())

        if self.print_message:
            print('Finish LLAMA')

        return loss
        # return {"loss": loss}
    
    def save_model(self):
        torch.save(self.state_dict(), self.save_model_path)

    def load_model(self):
        torch.load(self.load_model_path)



# print(my_model.device)
tokenizer = AutoTokenizer.from_pretrained("/data1/cchuan/data/weight/xlmr/")
# 填充在右侧进行
tokenizer.padding_side = "right"

texts = ['hello world, my name is cc!', 'i am heat', 'I love math', 'sduhgkuhwruhfsduhwiuehfiqeughfwiudhf123s1 23123i12 123 123 123 34 34太多分 rdsg 4t dfg erg rg 34r g4t rs3 243t udhf']
answers = ['hello, i am going to do my home work now! So do not contact me any more!',
           'hi, how are you?',
           'I hate English',
           '123 123 123 123 12 3123123123123 12 312 123 123 123 123 123 123 123 123 123 123 123 123 '
           ]

encoded_data = tokenizer(
    texts,
    padding="max_length",
    return_tensors="pt",
    truncation=True,
    max_length=128  # 显式指定最大长度
)

# print(encoded_data)
# sample = {'text': encoded_data, 'answer': answers}
def read_data():
    data_list = []

    data_path = '/data1/cchuan/data/mllm/clean_data1.json'

    with open(data_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            data_list.append(data)

    # 现在，data_list 包含了所有的 JSON 对象

    samples = {'prompt':[], 'completion': []}

    for item in data_list[0]['train'][: 3]:
        samples['prompt'].append(item['input'])
        samples['completion'].append(item['output'])
    
    print('finish reading')

    return samples
def solve_data(samples):
    encode_tokenizer_path = '/data1/cchuan/data/weight/xlmr/'

    encode_tokenizer = AutoTokenizer.from_pretrained(encode_tokenizer_path)
    encode_tokenizer.padding_side = 'right'
    encode_max_length = 128


    print('finish loading')

    encoded_data = encode_tokenizer(
        samples['prompt'],
        padding="max_length",
        return_tensors="pt",
        truncation=True,
        max_length=encode_max_length  # 显式指定最大长度
    )

    print('finish tokenize')

    data = {'text': encoded_data, 'answer': samples['completion']}

    return data

sample = read_data()

sample = solve_data(sample)

my_model = GPT()

my_model.to_device(GPU_DEVICE)

loss = my_model(sample)

print(loss.item())

# my_model.save_model()

# my_model.to("cpu")

# # 清空 GPU 缓存内存
# torch.cuda.empty_cache()

# print('save@!')

# new_model = GPT()
# new_model.load_model()

# print('load again!')

# # print(out)
