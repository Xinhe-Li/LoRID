import os
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any

from utils.file_utils import *
from utils.logger import LoggerHandler
from evaluation.grader import math_equal
from evaluation.parser import extract_answer
from hparams.prediction_args import prediction_args


logger_handler = LoggerHandler()
logger = logger_handler.get_logger()


def preprocess_intuitive_reasoner(
    dataset: List[Dict[str, Any]], dataset_name: str
) -> str:
    save_name = f"{dataset_name}_ir_test"
    save_path = f"save/data/{save_name}.json"
    dataset_info_path = "save/data/dataset_info.json"
    examples = []
    for item in tqdm(dataset):
        for _ in range(prediction_args.max_iterations):
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
    save_name = f"{dataset_name}_kg_test"
    save_path = f"save/data/{save_name}.json"
    dataset_info_path = "save/data/dataset_info.json"
    examples = []
    for item in tqdm(dataset):
        for _ in range(prediction_args.max_iterations):
            instruction = "Please output knowledge needed to solve the MATH question step by step."
            input_str = f"Question: {item['question']}\n\nKnowledge: "
            output_str = ""
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


def preprocess_deep_reasoner(
    dataset: List[Dict[str, Any]],
    knowledge_results: List[Dict[str, Any]],
    dataset_name: str,
) -> str:
    save_name = f"{dataset_name}_dr_test"
    save_path = f"save/data/{save_name}.json"
    dataset_info_path = "save/data/dataset_info.json"
    examples, index = [], 0
    for item in tqdm(dataset):
        for _ in range(prediction_args.max_iterations):
            instruction = "Please reason step by step based on the given MATH question and knowledge, and output the final answer."
            input_str = f"Question: {item['question']}\n\nKnowledge: {knowledge_results[index]['predict']}\n\nRationale: "
            if dataset_name == "gsm8k":
                output_str = item["rationale"] + f"\n--> {item['answer']} END"
            elif dataset_name == "math":
                output_str = item["rationale"]
            else:
                raise ValueError("Unknown dataset name")
            examples.append(
                {"instruction": instruction, "input": input_str, "output": output_str}
            )
            index += 1
    write_json(save_path, examples)

    if not os.path.isfile(dataset_info_path):
        write_json(dataset_info_path, {})
    dataset_info = read_json(dataset_info_path)
    dataset_info[save_name] = {"file_name": f"{save_name}.json"}
    write_json(dataset_info_path, dataset_info)
    return save_name


def run_prediction(
    model_name_or_path: str,
    dataset_name: str,
    output_dir: str,
    temperature: float,
    top_p: float,
) -> None:
    command = f"""CUDA_VISIBLE_DEVICES={prediction_args.device} python LLaMA-Factory/scripts/vllm_infer.py \
    --model_name_or_path {model_name_or_path} \
    --dataset {dataset_name} \
    --dataset_dir save/data \
    --template {prediction_args.template} \
    --output_dir {output_dir} \
    --temperature {temperature} \
    --top_p {top_p}"""

    os.system(command)


def evaluate(
    intuitive_results: List[Dict[str, Any]], deep_results: List[Dict[str, Any]]
) -> tuple[List[int], List[int]]:
    scores_list, iterations_list = [], []
    examples_num = len(intuitive_results) // prediction_args.max_iterations
    for i in tqdm(range(examples_num)):
        correct_label = 0
        label_answer = extract_answer(
            intuitive_results[i * 20]["label"], prediction_args.dataset
        )
        intuitive_results_set, deep_results_set = set(), set()
        for j in range(prediction_args.max_iterations):
            intuitive_results_set.add(
                extract_answer(
                    intuitive_results[i * 20 + j]["predict"], prediction_args.dataset
                )
            )
            deep_results_set.add(
                extract_answer(
                    deep_results[i * 20 + j]["predict"], prediction_args.dataset
                )
            )
            intuitive_results_set = {x for x in intuitive_results_set if x != ""}
            deep_results_set = {x for x in deep_results_set if x != ""}
            predict_answer_set = intuitive_results_set & deep_results_set
            if predict_answer_set:
                for predict_answer in list(predict_answer_set):
                    if math_equal(predict_answer, label_answer):
                        correct_label = 1
                break
        scores_list.append(correct_label)
        iterations_list.append(j + 1)
    return scores_list, iterations_list


def run() -> None:
    logger.info("Start...")
    dataset = read_jsonl(prediction_args.dataset_path)

    # intuitive reasoner
    logger.info("Intuitive Reasoner prediction.")
    dataset_name_ir = preprocess_intuitive_reasoner(dataset, prediction_args.dataset)
    run_prediction(
        model_name_or_path=prediction_args.model_name_or_path_ir,
        dataset_name=dataset_name_ir,
        output_dir=prediction_args.output_dir_ir,
        temperature=prediction_args.temperature_ir,
        top_p=prediction_args.top_p_ir,
    )

    # knowledge generator
    logger.info("Knowledge Generator prediction.")
    dataset_name_kg = preprocess_knowledge_generator(dataset, prediction_args.dataset)
    run_prediction(
        model_name_or_path=prediction_args.model_name_or_path_kg,
        dataset_name=dataset_name_kg,
        output_dir=prediction_args.output_dir_kg,
        temperature=prediction_args.temperature_kg,
        top_p=prediction_args.top_p_kg,
    )

    # deep reasoner
    logger.info("Deep Reasoner prediction.")
    knowledge_results = read_jsonl(
        f"{prediction_args.output_dir_kg}/generated_predictions.jsonl"
    )
    dataset_name_dr = preprocess_deep_reasoner(
        dataset, knowledge_results, prediction_args.dataset
    )
    run_prediction(
        model_name_or_path=prediction_args.model_name_or_path_dr,
        dataset_name=dataset_name_dr,
        output_dir=prediction_args.output_dir_dr,
        temperature=prediction_args.temperature_dr,
        top_p=prediction_args.top_p_dr,
    )

    # Evaluate
    logger.info("Evaluation.")
    intuitive_results = read_jsonl(
        f"{prediction_args.output_dir_ir}/generated_predictions.jsonl"
    )
    deep_results = read_jsonl(
        f"{prediction_args.output_dir_dr}/generated_predictions.jsonl"
    )
    scores, iterations = evaluate(intuitive_results, deep_results)
    logger.info(f"Accuracy: {np.mean(scores)}")
    logger.info(f"Average iterations: {np.mean(iterations)}")

    logger.info("End.")


if __name__ == "__main__":
    run()
