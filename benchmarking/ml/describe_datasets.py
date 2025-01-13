from pathlib import Path
import pandas as pd


subdir_stems_to_consider = [  # optionally comment out any of these
    "classification_mixed",
    "classification_numerical",
    "regression_mixed",
    "regression_numerical",
]


curr_dir = Path(__file__).resolve().parent

datasets_dir = curr_dir / "datasets"
subdirs = sorted([x for x in datasets_dir.iterdir() if x.is_dir()])
subdirs = [x for x in subdirs if x.stem in subdir_stems_to_consider]


output_path = curr_dir / "results" / "dataset_descriptions.csv"


output = {
    "id": [],
    "file_name": [],
    "target": [],
    "n_features": [],
    "n_examples": [],
}


for subdir in subdirs:
    for file in sorted(list(subdir.iterdir())):
        full_file_name = file.stem
        id = full_file_name.split("_")[0]
        file_name = "_".join(full_file_name.split("_")[1:])

        df = pd.read_csv(file)
        target = df.columns[0]

        n_features = df.shape[1] - 1
        n_examples = df.shape[0]

        output["id"].append(id)
        output["file_name"].append(file_name)
        output["target"].append(target)
        output["n_features"].append(n_features)
        output["n_examples"].append(n_examples)


results_df = pd.DataFrame(output)
results_df.to_csv(output_path, index=False)
