#!/bin/bash

python src/train.py \
    --module knowledge_generator \
    --dataset math \
    --dataset_path datasets/math/train.jsonl \
    --device 0 \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --lora_rank 512 \
    --lora_alpha 1024 \
    --template mistral \
    --output_dir_adapter save/adapter/math/mistral/knowledge_generator \
    --output_dir_model save/model/math/mistral/knowledge_generator \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-05 \
    --num_train_epochs 5.0 \
    --warmup_ratio 0.03