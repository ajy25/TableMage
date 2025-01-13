from metagpt.roles.di.data_interpreter import DataInterpreter
from pathlib import Path
import pandas as pd
import fire
from metagpt.logs import logger
import sys
import time


logger.add(sys.stdout, level="INFO")

dataset_name_to_filestem = {
    "Titanic": "titanic",
    "House Prices": "house_prices",
    "Wine": "wine",
    "Breast Cancer": "breast_cancer",
    "Credit": "credit",
    "Abalone": "abalone",
    "Baseball": "baseball",
    "Auto MPG": "autompg",
    "Healthcare": "simulated_healthcare",
    "Iris": "iris",
}


subset = []

curr_dir = Path(__file__).resolve().parent
output_dir = curr_dir / "results" / "di_gpt4o"
output_dir.mkdir(exist_ok=True, parents=True)
datasets_dir = curr_dir / "datasets"

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
    print(f"Generating answers for {dataset_name} dataset using DataInterpreter...")

    questions_subset_df = questions_df[questions_df["Dataset"] == dataset_name]

    n_questions = len(questions_subset_df)

    dataset_df_path = (
        curr_dir / "datasets" / f"{dataset_name_to_filestem[dataset_name]}.csv"
    )

    prompt = """\
    I am providing you with a dataset located at path {file_path}. \
    I will then give you instructions for analyzing the dataset.

    Here are some rules: \
    (1) Immediately split the dataset into 80/20 train/test sets using sklearn’s train_test_split function, with random seed 42. \
    (2) If you are explicitly asked to transform the dataset (e.g., scaling, imputation, feature engineering), keep the changes for future questions. \
    (3) When transforming data (e.g., feature scaling), always fit on the train dataset and transform the test dataset based on the train dataset. \
    (4) When fitting models, always fit on the train dataset and predict on the test dataset. \
    (5) For exploratory analysis (e.g., statistical testing, summary statistics), always consider the entire dataset. \
    (6) Temporarily drop rows with missing values in variables of interest prior to each analysis step. \
    (7) Return a sentence for each query describing your findings, round numeric answers to 3 decimal places. \
    (8) Use significance level 0.05 for statistical tests. \

    {task}
    """

    mi = DataInterpreter()

    for i in range(1, n_questions + 1):
        question_id = questions_subset_df[questions_subset_df["Question Order"] == i][
            "Question ID"
        ].values[0]

        question_ids.append(question_id)

        question = questions_subset_df[questions_subset_df["Question Order"] == i][
            "Question"
        ].values[0]

        try:

            async def main():
                mi = DataInterpreter()
                result = await mi.run(
                    prompt.format(
                        file_path=dataset_df_path,
                        task=question,
                    )
                )
                return str(result)

            answer = fire.Fire(main)
            unformatted_answers.append(answer)
        except Exception as e:
            answer = f"Error occurred: {e}. No answer generated."
            unformatted_answers.append(answer)

        print(f"\n\n\nQuestion ID: {i}\nQuestion: {question}\nAnswer: {answer}\n\n\n")

        time.sleep(1)


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
