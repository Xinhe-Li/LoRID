import os
import json
from openai import OpenAI


def use_gpt(role: str, query: str, gpt_histories_path: str, temperature: float) -> str:
    api_base = os.getenv("OPENAI_API_BASE", "")
    api_key = os.getenv("OPENAI_API_KEY", "")

    client = OpenAI(base_url=api_base, api_key=api_key)

    rsp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": role},
            {"role": "user", "content": query},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    response = rsp.choices[0].message.content

    with open(gpt_histories_path, "a+", encoding="utf8") as wf:
        data = {"query": query, "response": response}
        wf.write(json.dumps(data) + "\n")

    return response
