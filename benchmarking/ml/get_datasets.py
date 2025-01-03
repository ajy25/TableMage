import openml
import pandas as pd
from pathlib import Path

curr_path = Path(__file__).parent.resolve()

# read in the OpenML API key
api_key_path = curr_path / "api_key.txt"

openml.config.apikey = api_key_path.read_text().strip()

task_to_suite_id = {
    "regression_numerical": 336,
    "classification_numerical": 337,
    "regression_mixed": 335,
    "classification_mixed": 334,
}

for task in task_to_suite_id.keys():
    save_path = curr_path / "datasets" / task
    save_path.mkdir(exist_ok=True)

    SUITE_ID = task_to_suite_id[task]

    benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite

    for task_id in benchmark_suite.tasks:  # iterate over all tasks
        task = openml.tasks.get_task(task_id)  # download the OpenML task
        dataset = task.get_dataset()
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )
        X: pd.DataFrame = X
        y: pd.Series = y

        if len(X) > 10000 or X.shape[1] > 100:
            print(f"Skipping task {task_id} due to large dataset size")
            continue

        print(dataset.name)
        print(task)
        print(X.shape)
        print(X.columns)
        print(y.shape)
        print(y.name)

        # concatenate X and y
        df = pd.concat([y, X], axis=1)
        assert df.shape[1] == X.shape[1] + 1

        # excessively ensure that the target column is the first column
        df = df[[y.name] + [col for col in X.columns]]
        assert df.columns[0] == y.name

        # save to datasets/task folder
        df.to_csv(save_path / f"{task_id}_{dataset.name}.csv", index=False)
