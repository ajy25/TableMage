from pathlib import Path
import sys
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent.parent

sys.path.append(str(repo_root))

curr_dir = Path(__file__).resolve().parent

from tablemage.agents import ConversationalAgent, options

options.set_llm(llm_type="groq", model_name="llama-3.1-8b-instant", temperature=0.1)

import json
from utils.utils import read_jsonl

output_path = curr_dir / "data" / "tm-output.jsonl"
datasets_dir = curr_dir / "data" / "da-dev-tables"

questions = read_jsonl(file_name=curr_dir / "data" / "da-dev-questions.jsonl")

answers = []

# iterate over the questions and generate the answers
prev_dataset_name = None
agent = None

for question in questions:
    question_id = question["id"]
    question_text = question["question"]
    file_name = question["file_name"]

    df = pd.read_csv(
        datasets_dir / file_name,
    )

    if prev_dataset_name != file_name:
        agent = ConversationalAgent(df=df, test_size=0.2)

    try:
        response = agent.chat(message=question_text)
    except Exception as e:
        response = "Error."

    print(
        f"Number: {question_id}\n"
        f"Question: {question_text}\n"
        f"Answer: {response}\n"
    )

    answers.append({"id": question_id, "answer": str(response)})

# save the answers to a jsonl file
with open(output_path, "w") as f:
    for answer in answers:
        f.write(json.dumps(answer) + "\n")
