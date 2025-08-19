import os
import json
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Any

from utils.file_utils import *
from utils.gpt_utils import use_gpt
from utils.logger import LoggerHandler
from hparams.dataset_args import dataset_args

root_dir = os.getcwd()
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
gpt_histories_path = os.path.join(
    root_dir, f"save/call_histories/history_{current_time}.jsonl"
)

knowledge_prompt_path = os.path.join(root_dir, f"prompt/knowledge.txt")

logger_handler = LoggerHandler()
logger = logger_handler.get_logger()


def generate_knowledge(example: Dict[str, Any]) -> None:
    with open(knowledge_prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    input_str = f"Question: {example['question']}\nRationale: {example['rationale']}\nKnowledge: "
    prompt = prompt_template.replace("<input>", input_str)
    role = "You are a thoughtful and logical MATH teacher designed to output JSON."

    while 1:
        try:
            response = use_gpt(
                role=role,
                query=prompt,
                gpt_histories_path=gpt_histories_path,
                temperature=0.2,
            )
            response_list = json.loads(response)["output"]
            example["knowledge"] = response_list
            break
        except Exception as e:
            logger.warning(f"<{example['id']}> {type(e).__name__}: {e}")
    return example


def run() -> None:
    datasets = read_jsonl(dataset_args.raw_dataset_path)
    for item in tqdm(datasets):
        new_item = generate_knowledge(item)
        with open(dataset_args.dataset_path, "a+", encoding="utf8") as wf:
            wf.write(json.dumps(new_item) + "\n")


if __name__ == "__main__":
    run()
