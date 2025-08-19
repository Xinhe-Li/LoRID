import argparse


# training args
training_parser = argparse.ArgumentParser(
    description="Parser For Arguments",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
training_parser.add_argument(
    "--module",
    dest="module",
    default="intuitive_reasoner",
    type=str,
    choices=["intuitive_reasoner", "knowledge_generator", "deep_reasoner"],
    help="Type of LoRA module (intuitive_reasoner, knowledge_generator, or deep_reasoner)",
)
training_parser.add_argument(
    "--dataset",
    dest="dataset",
    default="gsm8k",
    type=str,
    choices=["gsm8k", "math"],
    help="Dataset name (gsm8k or math)",
)
training_parser.add_argument(
    "--dataset_path",
    dest="dataset_path",
    default="",
    type=str,
    help="Path to the training dataset with knowledge",
)
training_parser.add_argument(
    "--device", dest="device", default=0, type=int, help="Gpu id"
)
training_parser.add_argument(
    "--model_name_or_path",
    dest="model_name_or_path",
    default="",
    type=str,
    help="Path to the base model",
)
training_parser.add_argument(
    "--lora_rank", dest="lora_rank", default=512, type=int, help="LoRA rank"
)
training_parser.add_argument(
    "--lora_alpha", dest="lora_alpha", default=1024, type=int, help="LoRA alpha"
)
training_parser.add_argument(
    "--template",
    dest="template",
    default="llama2",
    type=str,
    choices=["llama2", "llama3", "mistral", "qwen", "deepseek"],
    help="Type of base model (llama2, llama3, mistral, qwen, or deepseek)",
)
training_parser.add_argument(
    "--output_dir_adapter",
    dest="output_dir_adapter",
    default="",
    type=str,
    help="Path to the output directory of LoRA adapter",
)
training_parser.add_argument(
    "--output_dir_model",
    dest="output_dir_model",
    default="",
    type=str,
    help="Path to the output directory of merged model",
)
training_parser.add_argument(
    "--per_device_train_batch_size",
    dest="per_device_train_batch_size",
    default=4,
    type=int,
    help="Batch size for training on each device",
)
training_parser.add_argument(
    "--gradient_accumulation_steps",
    dest="gradient_accumulation_steps",
    default=4,
    type=int,
    help="Gradient accumulation steps",
)
training_parser.add_argument(
    "--learning_rate",
    dest="learning_rate",
    default=5e-05,
    type=float,
    help="Learning rate",
)
training_parser.add_argument(
    "--num_train_epochs",
    dest="num_train_epochs",
    default=5.0,
    type=float,
    help="The number of training epochs",
)
training_parser.add_argument(
    "--warmup_ratio", dest="warmup_ratio", default=0.03, type=float, help="Warmup ratio"
)
training_args = training_parser.parse_args()
