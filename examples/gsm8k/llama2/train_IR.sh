#!/bin/bash

python src/train.py \
    --module intuitive_reasoner \
    --dataset gsm8k \
    --dataset_path datasets/gsm8k/train.jsonl \
    --device 0 \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --lora_rank 512 \
    --lora_alpha 1024 \
    --template llama2 \
    --output_dir_adapter save/adapter/gsm8k/llama2/intuitive_reasoner \
    --output_dir_model save/model/gsm8k/llama2/intuitive_reasoner \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-05 \
    --num_train_epochs 5.0 \
    --warmup_ratio 0.03