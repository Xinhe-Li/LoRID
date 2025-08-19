import argparse


# prediction args
prediction_parser = argparse.ArgumentParser(
    description="Parser For Arguments",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
prediction_parser.add_argument(
    "--dataset",
    dest="dataset",
    default="gsm8k",
    type=str,
    choices=["gsm8k", "math"],
    help="Dataset name (gsm8k or math)",
)
prediction_parser.add_argument(
    "--dataset_path",
    dest="dataset_path",
    default="",
    type=str,
    help="Path to the test dataset",
)
prediction_parser.add_argument(
    "--device", dest="device", default=0, type=int, help="Gpu id"
)
prediction_parser.add_argument(
    "--model_name_or_path_ir",
    dest="model_name_or_path_ir",
    default="",
    type=str,
    help="Path to the LoRA adapter of intuitive reasoner",
)
prediction_parser.add_argument(
    "--model_name_or_path_kg",
    dest="model_name_or_path_kg",
    default="",
    type=str,
    help="Path to the LoRA adapter of knowledge generator",
)
prediction_parser.add_argument(
    "--model_name_or_path_dr",
    dest="model_name_or_path_dr",
    default="",
    type=str,
    help="Path to the LoRA adapter of deep reasoner",
)
prediction_parser.add_argument(
    "--template",
    dest="template",
    default="llama2",
    type=str,
    choices=["llama2", "llama3", "mistral", "qwen", "deepseek"],
    help="Type of base model (llama2, llama3, mistral, qwen, or deepseek)",
)
prediction_parser.add_argument(
    "--output_dir_ir",
    dest="output_dir_ir",
    default="",
    type=str,
    help="Path to the output directory of intuitive reasoner",
)
prediction_parser.add_argument(
    "--output_dir_kg",
    dest="output_dir_kg",
    default="",
    type=str,
    help="Path to the output directory of knowledge generator",
)
prediction_parser.add_argument(
    "--output_dir_dr",
    dest="output_dir_dr",
    default="",
    type=str,
    help="Path to the output directory of deep reasoner",
)
prediction_parser.add_argument(
    "--temperature_ir",
    dest="temperature_ir",
    default=1.50,
    type=float,
    help="Temperature for intuitive reasoner",
)
prediction_parser.add_argument(
    "--temperature_kg",
    dest="temperature_kg",
    default=1.50,
    type=float,
    help="Temperature for knowledge generator",
)
prediction_parser.add_argument(
    "--temperature_dr",
    dest="temperature_dr",
    default=1.50,
    type=float,
    help="Temperature for deep reasoner",
)
prediction_parser.add_argument(
    "--top_p_ir",
    dest="top_p_ir",
    default=0.90,
    type=float,
    help="Top-p for intuitive reasoner",
)
prediction_parser.add_argument(
    "--top_p_kg",
    dest="top_p_kg",
    default=0.90,
    type=float,
    help="Top-p for knowledge generator",
)
prediction_parser.add_argument(
    "--top_p_dr",
    dest="top_p_dr",
    default=0.90,
    type=float,
    help="Top-p for deep reasoner",
)
prediction_parser.add_argument(
    "--max_iterations",
    dest="max_iterations",
    default=20,
    type=int,
    help="Max iterations",
)
prediction_args = prediction_parser.parse_args()
