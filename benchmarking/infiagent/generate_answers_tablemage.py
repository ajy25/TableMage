from pathlib import Path
import sys
import pandas as pd
import time

repo_root = Path(__file__).resolve().parent.parent.parent

sys.path.append(str(repo_root))

curr_dir = Path(__file__).resolve().parent

import tablemage as tm

tm.use_agents()
tm.agents.options.set_llm(llm_type="openai", model_name="gpt-4o", temperature=0.0)

import json
from utils.utils import read_jsonl

output_path = curr_dir / "data" / "tm-output.jsonl"
datasets_dir = curr_dir / "data" / "da-dev-tables"
questions = read_jsonl(file_name=curr_dir / "data" / "da-dev-questions.jsonl")


AGENT_SYSTEM_PROMPT = """\
You will be asked a series of questions about the data in the table. \
You have access to tools to analyze the data.
Use the tools to answer the questions.
Respond with ONLY the answer to the question. No other words are necessary.
Round your answers to 4 decimal places.
"""


answers = []

# iterate over the questions and generate the answers
prev_dataset_name = None
agent = None


for question in questions:
    question_id = question["id"]
    question_text = question["question"]
    file_name = question["file_name"]

    # if we have moved on to a new dataset, save the answers to the file
    # make sure to append, not overwrite
    if prev_dataset_name != file_name and prev_dataset_name is not None:
        print(f"Saving answers for {prev_dataset_name} to {output_path}.")
        with open(output_path, "a") as f:
            for answer in answers:
                f.write(json.dumps(answer) + "\n")
        answers = []

    df = pd.read_csv(
        datasets_dir / file_name,
    )
    # if the first column is unnamed, drop it
    if df.columns[0] == "Unnamed: 0":
        df = df.drop(columns="Unnamed: 0")

    if prev_dataset_name != file_name:
        agent = tm.agents.ConversationalAgent(
            df=df, test_size=0.2, system_prompt=AGENT_SYSTEM_PROMPT, python_only=False
        )
    try:
        response = agent.chat(message=question_text)
    except Exception as e:
        response = f"Error: {e}."

    print("\n\n\n")
    print("-" * 80)

    print(
        f"Number: {question_id}\n"
        f"Question: {question_text}\n"
        f"Answer: {response}\n"
    )

    answers.append({"id": question_id, "answer": str(response)})

    prev_dataset_name = file_name

    print("-" * 80)
    print("\n\n\n")
    # pause for a bit to avoid rate limiting
    time.sleep(10)
