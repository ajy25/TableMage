from metagpt.roles.di.data_interpreter import DataInterpreter
from pathlib import Path
import pandas as pd
import fire
from metagpt.logs import logger
import sys

logger.add(sys.stdout, level="INFO")

subdir_stems_to_consider = [
    "classification_mixed",
    "classification_numerical",
    "regression_mixed",
    "regression_numerical",
]


curr_dir = Path(__file__).resolve().parent
output_dir = curr_dir / "results" / "di_gpt4o_mini_results"
output_dir.mkdir(exist_ok=True)
datasets_dir = curr_dir / "datasets"
subdirs = sorted([x for x in datasets_dir.iterdir() if x.is_dir()])
subdirs = [x for x in subdirs if x.stem in subdir_stems_to_consider]


for subdir in subdirs:
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
    }

    for file in sorted(list(subdir.iterdir())):
        file_path = file

        df = pd.read_csv(file_path)
        target = df.columns[0]

        prompt = """\
        The dataset is at {file_path}.
        You must perform a 60/40 train/test split using sklearn’s train_test_split function, with random seed 42.

        Predict the variable `{target}` with machine learning {task}. \
        Report the test {metric} of the best possible model you can train. \
        Prior to training, consider employing feature selection and/or feature scaling/transformation to improve model performance. \
        Only report the test {metric} value, rounded to 3 decimal points.
        """

        async def main():
            mi = DataInterpreter()
            result = await mi.run(
                prompt.format(
                    file_path=file_path,
                    target=target,
                    task=task,
                    metric=metric,
                )
            )
            return str(result)

        output = fire.Fire(main)

        results_dict["file_name"].append(file.stem)
        results_dict["unformatted_answer"].append(output)

    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(output_path, index=False)
