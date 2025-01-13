import pathlib
import sys
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(parent_dir))


import tablemage as tm


@pytest.fixture
def setup_data():
    df_simple = pd.DataFrame(
        {
            "binary_var": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
            "categorical_var_1": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
            "categorical_var_2": ["X", "Y", "Z", "X", "Y", "Z", "X", "Y", "Z", "X"],
            "numeric_var_1": [5.2, 3.7, 8.1, 2.5, 6.9, 4.3, 7.8, 1.1, 3.4, 5.7],
            "numeric_var_2": [1.2, 3.4, 5.6, 7.8, 9.1, 2.3, 4.5, 6.7, 8.9, 0.1],
        }
    )
    df_simple_train, df_simple_test = train_test_split(
        df_simple, test_size=0.5, random_state=42
    )
    return {
        "df_simple_train": df_simple_train,
        "df_simple_test": df_simple_test,
    }


def test_fs_simple(setup_data):
    """Tests feature selection mechanism for prediction with ml models"""
    analyzer = tm.Analyzer(
        setup_data["df_simple_train"], setup_data["df_simple_test"], verbose=False
    )

    report = analyzer.classify(
        models=[tm.ml.LinearC(type="l2", n_trials=5, name="simple_tester")],
        target="binary_var",
        predictors=[
            "categorical_var_1",
            "categorical_var_2",
            "numeric_var_1",
            "numeric_var_2",
        ],
        feature_selectors=[tm.fs.KBestFSC("chi2", 2)],
    )
    assert report.model("simple_tester").sklearn_estimator().n_features_in_ == 2

    report = analyzer.classify(
        models=[
            tm.ml.LinearC(
                type="l2",
                n_trials=5,
                name="simple_tester",
                feature_selectors=[tm.fs.KBestFSC("chi2", 2)],
            )
        ],
        target="binary_var",
        predictors=[
            "categorical_var_1",
            "categorical_var_2",
            "numeric_var_1",
            "numeric_var_2",
        ],
    )
    assert report.model("simple_tester").sklearn_estimator().n_features_in_ == 2

    report = analyzer.classify(
        models=[
            tm.ml.LinearC(
                type="l2",
                n_trials=5,
                name="simple_tester",
                feature_selectors=[tm.fs.KBestFSC("chi2", 2)],
            ),
            tm.ml.LinearC(
                type="l2",
                n_trials=5,
                name="simple_tester_2",
            ),
        ],
        target="binary_var",
        predictors=[
            "categorical_var_1",
            "categorical_var_2",
            "numeric_var_1",
            "numeric_var_2",
        ],
        feature_selectors=[tm.fs.KBestFSC("chi2", 4)],
    )
    assert report.model("simple_tester").sklearn_estimator().n_features_in_ == 2
    assert report.model("simple_tester_2").sklearn_estimator().n_features_in_ == 4

    report = analyzer.regress(
        models=[
            tm.ml.LinearR(
                type="ols",
                name="simple_tester",
                feature_selectors=[tm.fs.KBestFSR("r_regression", 2)],
            ),
            tm.ml.LinearR(
                type="ols",
                name="simple_tester_2",
            ),
        ],
        target="numeric_var_2",
        predictors=[
            "categorical_var_1",
            "categorical_var_2",
            "binary_var",
            "numeric_var_1",
        ],
        feature_selectors=[tm.fs.KBestFSR("f_regression", 3)],
    )
    assert report.model("simple_tester").sklearn_estimator().n_features_in_ == 2
    assert report.model("simple_tester_2").sklearn_estimator().n_features_in_ == 3
    assert len(report.model("simple_tester").predictors()) == 2
    assert len(report.model("simple_tester_2").predictors()) == 3
    assert report.model("simple_tester")._best_estimator.n_features_in_ == 2
    assert report.model("simple_tester_2")._best_estimator.n_features_in_ == 3
