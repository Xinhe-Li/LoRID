#!/bin/bash

python src/predict.py \
    --dataset gsm8k \
    --dataset_path datasets/gsm8k/test.jsonl \
    --device 0 \
    --model_name_or_path_ir save/model/gsm8k/llama2/intuitive_reasoner \
    --model_name_or_path_kg save/model/gsm8k/llama2/knowledge_generator \
    --model_name_or_path_dr save/model/gsm8k/llama2/deep_reasoner \
    --template llama2 \
    --output_dir_ir save/predict/gsm8k/llama2/intuitive_reasoner \
    --output_dir_kg save/predict/gsm8k/llama2/knowledge_generator \
    --output_dir_dr save/predict/gsm8k/llama2/deep_reasoner \
    --temperature_ir 1.50 \
    --temperature_kg 1.50 \
    --temperature_dr 1.50 \
    --top_p_ir 0.90 \
    --top_p_kg 0.90 \
    --top_p_dr 0.90 \
    --max_iterations 20