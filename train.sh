#!/bin/bash

# pip install -r requirements.txt

chmod +x ./s5cmd

DISTRIBUTED_ARGS="--nproc_per_node $SM_NUM_GPUS --nnodes $NODE_NUMBER --node_rank $NODE_INDEX --master_addr $SM_MASTER_ADDR --master_port 12345"

torchrun ${DISTRIBUTED_ARGS} stanford_alpaca/train.py \
    --model_name_or_path /tmp/llama_pretrain/ \
    --data_path stanford_alpaca/alpaca_data.json \
    --bf16 True \
    --output_dir /tmp/llama_out \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --deepspeed stanford_alpaca/configs/default_offload_opt_param.json \
    --tf32 True \
    --load_best_model_at_end False \
    --model_max_length 512 \
    --cache_dir /tmp


if [ $? -eq 1 ]; then
    echo "Training script error, please check CloudWatch logs"
    exit 1
fi

