import os
import yaml
import random
from tqdm import tqdm
from typing import List, Dict, Any

from utils.file_utils import *
from hparams.training_args import training_args

random.seed(42)


def preprocess_intuitive_reasoner(
    dataset: List[Dict[str, Any]], dataset_name: str
) -> str:
    save_name = f"{dataset_name}_ir_train"
    save_path = f"save/data/{save_name}.json"
    dataset_info_path = "save/data/dataset_info.json"
    examples = []
    for item in tqdm(dataset):
        instruction = "Please answer this MATH question step by step."
        input_str = item["question"]
        if dataset_name == "gsm8k":
            output_str = item["rationale"] + f"\n--> {item['answer']} END"
        elif dataset_name == "math":
            output_str = item["rationale"]
        else:
            raise ValueError("Unknown dataset name")
        examples.append(
            {"instruction": instruction, "input": input_str, "output": output_str}
        )
    write_json(save_path, examples)

    if not os.path.isfile(dataset_info_path):
        write_json(dataset_info_path, {})
    dataset_info = read_json(dataset_info_path)
    dataset_info[save_name] = {"file_name": f"{save_name}.json"}
    write_json(dataset_info_path, dataset_info)
    return save_name


def preprocess_knowledge_generator(
    dataset: List[Dict[str, Any]], dataset_name: str
) -> str:
    save_name = f"{dataset_name}_kg_train"
    save_path = f"save/data/{save_name}.json"
    dataset_info_path = "save/data/dataset_info.json"
    examples = []
    for item in tqdm(dataset):
        instruction = (
            "Please output knowledge needed to solve the MATH question step by step."
        )
        input_str = f"Question: {item['question']}\n\nKnowledge: "
        output_str = "\n".join(item["knowledge"])
        examples.append(
            {"instruction": instruction, "input": input_str, "output": output_str}
        )
    write_json(save_path, examples)

    if not os.path.isfile(dataset_info_path):
        write_json(dataset_info_path, {})
    dataset_info = read_json(dataset_info_path)
    dataset_info[save_name] = {"file_name": f"{save_name}.json"}
    write_json(dataset_info_path, dataset_info)
    return save_name


def preprocess_deep_reasoner(dataset: List[Dict[str, Any]], dataset_name: str) -> str:
    save_name = f"{dataset_name}_dr_train"
    save_path = f"save/data/{save_name}.json"
    dataset_info_path = "save/data/dataset_info.json"
    examples = []
    for item in tqdm(dataset):
        instruction = "Please reason step by step based on the given MATH question and knowledge, and output the final answer."
        input_str = (
            f"Question: {item['question']}\n\nKnowledge: "
            + "\n".join(item["knowledge"])
            + "\n\nRationale: "
        )
        if dataset_name == "gsm8k":
            output_str = item["rationale"] + f"\n--> {item['answer']} END"
        elif dataset_name == "math":
            output_str = item["rationale"]
        else:
            raise ValueError("Unknown dataset name")
        examples.append(
            {"instruction": instruction, "input": input_str, "output": output_str}
        )
    write_json(save_path, examples)

    if not os.path.isfile(dataset_info_path):
        write_json(dataset_info_path, {})
    dataset_info = read_json(dataset_info_path)
    dataset_info[save_name] = {"file_name": f"{save_name}.json"}
    write_json(dataset_info_path, dataset_info)
    return save_name


def run_training(dataset_name: str) -> None:
    config_sections = [
        (
            "### model",
            {
                "model_name_or_path": training_args.model_name_or_path,
                "trust_remote_code": True,
            },
        ),
        (
            "### method",
            {
                "stage": "sft",
                "do_train": True,
                "finetuning_type": "lora",
                "lora_rank": training_args.lora_rank,
                "lora_alpha": training_args.lora_alpha,
                "lora_dropout": 0.1,
                "lora_target": "q_proj,v_proj",
            },
        ),
        (
            "### dataset",
            {
                "dataset": dataset_name,
                "dataset_dir": "save/data",
                "template": training_args.template,
                "overwrite_cache": True,
                "preprocessing_num_workers": 16,
            },
        ),
        (
            "### output",
            {
                "output_dir": training_args.output_dir_adapter,
                "logging_steps": 5,
                "save_strategy": "no",
                "plot_loss": True,
                "overwrite_output_dir": True,
            },
        ),
        (
            "### train",
            {
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "learning_rate": training_args.learning_rate,
                "num_train_epochs": training_args.num_train_epochs,
                "lr_scheduler_type": "cosine",
                "warmup_ratio": training_args.warmup_ratio,
                "bf16": True,
                "ddp_timeout": 180000000,
            },
        ),
    ]

    with open("scripts/train.yaml", "w+") as f:
        for comment, config in config_sections:
            f.write(f"{comment}\n")
            yaml_str = yaml.dump(
                config,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
            f.write(yaml_str)
            f.write("\n")

    os.system(
        f"CUDA_VISIBLE_DEVICES={training_args.device} llamafactory-cli train scripts/train.yaml"
    )


def run_merging() -> None:
    config_sections = [
        (
            "### model",
            {
                "model_name_or_path": training_args.model_name_or_path,
                "adapter_name_or_path": training_args.output_dir_adapter,
                "template": training_args.template,
                "finetuning_type": "lora",
                "trust_remote_code": True,
            },
        ),
        (
            "### export",
            {
                "export_dir": training_args.output_dir_model,
                "export_size": 10,
                "export_device": "cpu",
                "export_legacy_format": False,
            },
        ),
    ]

    with open("scripts/merge.yaml", "w+") as f:
        for comment, config in config_sections:
            f.write(f"{comment}\n")
            yaml_str = yaml.dump(
                config,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
            f.write(yaml_str)
            f.write("\n")

    os.system(
        f"CUDA_VISIBLE_DEVICES={training_args.device} llamafactory-cli export scripts/merge.yaml"
    )


def run() -> None:
    dataset = read_jsonl(training_args.dataset_path)
    if training_args.module == "intuitive_reasoner":
        dataset_name = preprocess_intuitive_reasoner(dataset, training_args.dataset)
    elif training_args.module == "knowledge_generator":
        dataset_name = preprocess_knowledge_generator(dataset, training_args.dataset)
    elif training_args.module == "deep_reasoner":
        dataset_name = preprocess_deep_reasoner(dataset, training_args.dataset)
    else:
        raise ValueError(
            "Please choose argument --module from [intuitive_reasoner, knowledge_generator, deep_reasoner]."
        )

    run_training(dataset_name)
    run_merging()


if __name__ == "__main__":
    run()
