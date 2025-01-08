from pathlib import Path
import sys
import pandas as pd
import time

repo_root = Path(__file__).resolve().parent.parent.parent

sys.path.append(str(repo_root))

curr_dir = Path(__file__).resolve().parent

import tablemage as tm

tm.use_agents()


model_name = "llama3.1_8b"

subdir_stems_to_consider = [  # optionally comment out any of these
    # "classification_mixed",
    # "classification_numerical",
    # "regression_mixed",
    "regression_numerical",
]


if model_name == "llama3.1_8b":
    tm.agents.options.set_llm(
        llm_type="groq", model_name="llama-3.1-8b-instant", temperature=0.1
    )
    output_dir = curr_dir / "results" / "tablemage_llama3.1_8b_test"

elif model_name == "llama3.3_70b":
    tm.agents.options.set_llm(
        llm_type="groq", model_name="llama-3.3-70b-versatile", temperature=0.1
    )
    output_dir = curr_dir / "results" / "tablemage_llama3.3_70b"

elif model_name == "gpt4o":
    tm.agents.options.set_llm(llm_type="openai", model_name="gpt-4o", temperature=0.1)
    output_dir = curr_dir / "results" / "tablemage_gpt4o"

elif model_name == "gpt4o_mini":
    tm.agents.options.set_llm(
        llm_type="openai", model_name="gpt-4o-mini", temperature=0.1
    )
    output_dir = curr_dir / "results" / "tablemage_gpt4o_mini"


output_dir.mkdir(exist_ok=True)
datasets_dir = curr_dir / "datasets"
subdirs = sorted([x for x in datasets_dir.iterdir() if x.is_dir()])
subdirs = [x for x in subdirs if x.stem in subdir_stems_to_consider]


for subdir in subdirs:
    print(f"Generating answers for {subdir.stem} dataset using TableMage...")

    if subdir.stem.split("_")[0] == "regression":
        metric = "RMSE"
        task = "regression"
    elif subdir.stem.split("_")[0] == "classification":
        metric = "AUROC"
        task = "classification"
    else:
        raise RuntimeError("Unknown dataset type")

    output_path = output_dir / f"{subdir.stem}_performances.csv"

    results_dict = {
        "file_name": [],
        "unformatted_answer": [],
        "time_elapsed": [],
    }

    for file in sorted(list(subdir.iterdir())):
        df = pd.read_csv(file)

        print(df.shape)

        start = time.time()

        target = df.columns[0]

        agent = tm.agents.ConversationalAgent(
            df=df,
            test_size=0.4,
            split_seed=42,
            tool_rag_top_k=5,
            tools_only=True,
            system_prompt="""\
You are a helpful assistant. \
You have access to tools for training and evaluating machine learning models. \
Use your tools to help the user train the best possible model for the given dataset.
""",
        )

        response = agent.chat(
            f"Predict the variable `{target}` with machine learning {task}. "
            f"Report the test {metric} of the best possible model you can train. "
            "Prior to training, consider employing feature selection and/or "
            "feature scaling/transformation to improve model performance. "
            f"Only report the test {metric} value, rounded to 3 decimal points."
        )

        end = time.time()
        elapsed = end - start

        print(
            "\n\n"
            + "-" * 80
            + "\n"
            + f"TableMage's answer for {file.stem}:\n{response}"
            + "\n"
            + "-" * 80
            + "\n\n"
        )

        results_dict["file_name"].append(file.stem)
        results_dict["unformatted_answer"].append(response)
        results_dict["time_elapsed"].append(elapsed)

    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(output_path, index=False)
