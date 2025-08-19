#!/bin/bash

python src/train.py \
    --module deep_reasoner \
    --dataset math \
    --dataset_path datasets/math/train.jsonl \
    --device 0 \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --lora_rank 512 \
    --lora_alpha 1024 \
    --template llama2 \
    --output_dir_adapter save/adapter/math/llama2/deep_reasoner \
    --output_dir_model save/model/math/llama2/deep_reasoner \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-05 \
    --num_train_epochs 5.0 \
    --warmup_ratio 0.03