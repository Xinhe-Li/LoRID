#!/bin/bash

python src/predict.py \
    --dataset math \
    --dataset_path datasets/math/test.jsonl \
    --device 0 \
    --model_name_or_path_ir save/model/math/llama3/intuitive_reasoner \
    --model_name_or_path_kg save/model/math/llama3/knowledge_generator \
    --model_name_or_path_dr save/model/math/llama3/deep_reasoner \
    --template llama3 \
    --output_dir_ir save/predict/math/llama3/intuitive_reasoner \
    --output_dir_kg save/predict/math/llama3/knowledge_generator \
    --output_dir_dr save/predict/math/llama3/deep_reasoner \
    --temperature_ir 1.50 \
    --temperature_kg 1.50 \
    --temperature_dr 1.50 \
    --top_p_ir 0.90 \
    --top_p_kg 0.90 \
    --top_p_dr 0.90 \
    --max_iterations 20