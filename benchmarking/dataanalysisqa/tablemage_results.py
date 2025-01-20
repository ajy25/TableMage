from pathlib import Path
import sys
import pandas as pd
import time

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(repo_root))
curr_dir = Path(__file__).resolve().parent

import tablemage as tm

tm.use_agents()


# set model name
model_name = "llama3.1_8b"

# set subset of datasets to generate answers for (must have already ran on all datasets)
# leave empty to generate answers for all datasets
subset = []


if model_name == "llama3.1_8b":
    tm.agents.options.set_llm(
        llm_type="groq", model_name="llama-3.1-8b-instant", temperature=0.1
    )
    output_dir = curr_dir / "results" / "tablemage_llama3.1_8b"
    delay = 5

elif model_name == "llama3.3_70b":
    tm.agents.options.set_llm(
        llm_type="groq", model_name="llama-3.3-70b-versatile", temperature=0.1
    )
    output_dir = curr_dir / "results" / "tablemage_llama3.3_70b"
    delay = 4

elif model_name == "gpt4o":
    tm.agents.options.set_llm(llm_type="openai", model_name="gpt-4o", temperature=0.1)
    output_dir = curr_dir / "results" / "tablemage_gpt4o"
    delay = 3

elif model_name == "gpt4o_mini":
    tm.agents.options.set_llm(
        llm_type="openai", model_name="gpt-4o-mini", temperature=0.1
    )
    output_dir = curr_dir / "results" / "tablemage_gpt4o_mini"
    delay = 5

output_dir.mkdir(exist_ok=True, parents=True)


dataset_name_to_filestem = {
    "Titanic": "titanic.csv",
    "House Prices": "house_prices.csv",
    "Wine": "wine.csv",
    "Breast Cancer": "breast_cancer.csv",
    "Credit": "credit.csv",
    "Abalone": "abalone.csv",
    "Baseball": "baseball.csv",
    "Auto MPG": "autompg.csv",
    "Healthcare": "simulated_healthcare.csv",
    "Iris": "iris.csv",
}


# read in the questions
questions_df = pd.read_csv(curr_dir / "dataanalysisqa.tsv", sep="\t")
results_df = None
if len(subset) != 0 and len(subset) < len(dataset_name_to_filestem):
    # try to read in full output
    results_df = pd.read_csv(output_dir / "answers.csv")
    dataset_name_to_filestem = {
        k: v for k, v in dataset_name_to_filestem.items() if k in subset
    }


question_ids = []
unformatted_answers = []


for dataset_name in dataset_name_to_filestem.keys():
    print(f"Generating answers for {dataset_name} dataset using TableMage...")

    questions_subset_df = questions_df[questions_df["Dataset"] == dataset_name]

    n_questions = len(questions_subset_df)

    dataset_df = pd.read_csv(
        curr_dir / "datasets" / (dataset_name_to_filestem[dataset_name])
    )
    agent = tm.agents.ConversationalAgent(
        df=dataset_df,
        test_size=0.2,
        split_seed=42,
        tool_rag=True,
        tool_rag_top_k=5,
        system_prompt="""\
You are a helpful assistant. \
You have access to tools for data analysis. \
Use your tools to help the user analyze the dataset. \
If you need to transform data, use your (non-Python) tools to do so. \
Round answers to 3 decimal places. \
""",
    )

    for i in range(1, n_questions + 1):
        question_id = questions_subset_df[questions_subset_df["Question Order"] == i][
            "Question ID"
        ].values[0]

        question_ids.append(question_id)

        question = questions_subset_df[questions_subset_df["Question Order"] == i][
            "Question"
        ].values[0]

        try:
            answer = agent.chat(question)
            unformatted_answers.append(answer)
        except Exception as e:
            answer = f"Error occurred: {e}. No answer generated."
            unformatted_answers.append(answer)

        print(f"\n\n\nQuestion ID: {i}\nQuestion: {question}\nAnswer: {answer}\n\n\n")

        # wait to avoid rate limiting
        time.sleep(delay)


if results_df is not None:
    # if we're working with a subset, update only the rows that have been updated
    for i, question_id in enumerate(question_ids):
        results_df.loc[
            results_df["Question ID"] == question_id, "Unformatted Answer"
        ] = unformatted_answers[i]

else:
    results_df = pd.DataFrame(
        {
            "Question ID": question_ids,
            "Unformatted Answer": unformatted_answers,
        }
    )

results_df.to_csv(output_dir / "answers.csv", index=False)
