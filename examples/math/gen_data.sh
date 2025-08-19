#!/bin/bash

export OPENAI_API_BASE="https://api.openai.com/v1"
export OPENAI_API_KEY="your_api_key"

python src/gen_data.py \
    --raw_dataset_path datasets/math/train_raw.jsonl \
    --dataset_path datasets/math/train.jsonl