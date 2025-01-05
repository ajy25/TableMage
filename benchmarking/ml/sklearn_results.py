from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, root_mean_squared_error
import numpy as np


subdir_stems_to_consider = [  # optionally comment out any of these
    "classification_mixed",
    "classification_numerical",
    "regression_mixed",
    "regression_numerical",
]


curr_dir = Path(__file__).resolve().parent

output_dir = curr_dir / "results" / "sklearn"
output_dir.mkdir(exist_ok=True)

datasets_dir = curr_dir / "datasets"
subdirs = sorted([x for x in datasets_dir.iterdir() if x.is_dir()])
subdirs = [x for x in subdirs if x.stem in subdir_stems_to_consider]


for subdir in subdirs:
    print(f"Generating answers for {subdir.stem} dataset using scikit-learn...")

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
        "linear": [],
        "random_forest": [],
    }

    for file in sorted(list(subdir.iterdir())):
        df = pd.read_csv(file)

        target = df.columns[0]

        print(target)

        y = df[target]
        X = df.drop(columns=[target])

        if task == "classification":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.4, random_state=42
            )

            linear = LogisticRegression(random_state=0)
            linear.fit(X_train, y_train)
            linear_preds = linear.predict(X_test)
            linear_score = roc_auc_score(y_test, linear_preds)

            random_forest = RandomForestClassifier(random_state=0)
            random_forest.fit(X_train, y_train)
            random_forest_preds = random_forest.predict(X_test)
            random_forest_score = roc_auc_score(y_test, random_forest_preds)

            results_dict["linear"].append(np.round(linear_score, 3))
            results_dict["random_forest"].append(np.round(random_forest_score, 3))

        elif task == "regression":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.4, random_state=42
            )

            linear = LinearRegression()
            linear.fit(X_train, y_train)
            linear_preds = linear.predict(X_test)
            linear_score = root_mean_squared_error(y_test, linear_preds)

            random_forest = RandomForestRegressor(random_state=0)
            random_forest.fit(X_train, y_train)
            random_forest_preds = random_forest.predict(X_test)
            random_forest_score = root_mean_squared_error(y_test, random_forest_preds)

            results_dict["linear"].append(np.round(linear_score, 3))
            results_dict["random_forest"].append(np.round(random_forest_score, 3))

        results_dict["file_name"].append(file.stem)

    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(output_path, index=False)
