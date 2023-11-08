export CUDA_VISIBLE_DEVICES=1,2,3,4,5

MODEL_SIZE=1M
NUM_GPUS=5
BATCH_SIZE_PER_GPU=8
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# 几个需要修改的地方
# model_name_or_path 模型的路径
# tokenizer_name tokenizer的名字
# train_file 训练数据的路径
# output_dir 输出的路径
accelerate launch \
    --mixed_precision fp16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage2.conf \
    \
    /home/cchuan/Project/open-instruct/open_instruct/finetune.py \
    \
    --use_flash_attn \
    --max_seq_length 256\
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 0.005 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 5 \
    --output_dir /data1/cchuan/model_weight/GPT_${MODEL_SIZE}/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1