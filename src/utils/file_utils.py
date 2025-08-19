import json
from typing import List, Dict, Any


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data_dict = []
    with open(file_path, "r", encoding="utf8") as file:
        for line in file:
            data = json.loads(line.strip())
            data_dict.append(data)
    return data_dict


def write_jsonl(file_path: str, data_dict: List[Dict[str, Any]]) -> None:
    with open(file_path, "w+", encoding="utf8") as file:
        for data in data_dict:
            file.write(json.dumps(data) + "\n")


def read_json(file_path: str) -> Dict[str, Any] | List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf8") as file:
        return json.load(file)


def write_json(file_path: str, data_dict: Dict[str, Any]) -> None:
    with open(file_path, "w+", encoding="utf8") as file:
        file.write(json.dumps(data_dict))
