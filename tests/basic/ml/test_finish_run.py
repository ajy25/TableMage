import pathlib
import sys
import pandas as pd
import pytest
from sklearn.linear_model import Ridge, LogisticRegression


parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(parent_dir))


import tablemage as tm

SAMPLE_SIZE = 100


@pytest.fixture
def setup_data():
    df_house = pd.read_csv(
        parent_dir / "demo" / "regression" / "house_price_data" / "data.csv"
    )
    df_house["ExterQual_binary"] = df_house["ExterQual"] == "TA"

    df_house = df_house[
        [
            "MSZoning",
            "ExterQual_binary",
            "YearBuilt",
            "LotArea",
            "OverallQual",
            "SalePrice",
        ]
    ]

    df_house_mini = df_house.sample(SAMPLE_SIZE, random_state=42)

    return {
        "df_house": df_house,
        "df_house_mini": df_house_mini,
    }


def test_regression_run_simple(setup_data):
    """Tests regression mechanism for prediction with ml models"""
    analyzer = tm.Analyzer(setup_data["df_house_mini"], test_size=0.4, verbose=False)
    report = analyzer.regress(
        models=[
            tm.ml.LinearR(
                type="l2",
                n_trials=1,
            ),
            tm.ml.CustomR(estimator=Ridge()),
        ],
        target="SalePrice",
        predictors=[
            "MSZoning",
            "ExterQual_binary",
            "LotArea",
            "OverallQual",
        ],
    )
    assert len(report.model("LinearR(l2)")._test_scorer._y_pred) == SAMPLE_SIZE * 0.4
    report._to_dict()


def test_regression_run_cv(setup_data):
    """Tests regression mechanism for prediction with ml models"""
    analyzer = tm.Analyzer(setup_data["df_house_mini"], test_size=0.4, verbose=False)
    report = analyzer.regress(
        models=[
            tm.ml.LinearR(
                type="l2",
                n_trials=1,
            ),
            tm.ml.CustomR(estimator=Ridge()),
        ],
        target="SalePrice",
        predictors=[
            "MSZoning",
            "ExterQual_binary",
            "LotArea",
            "OverallQual",
        ],
        outer_cv=2,
    )
    report._to_dict()


def test_regression_run_fs(setup_data):
    """Tests regression mechanism for prediction with ml models"""
    analyzer = tm.Analyzer(setup_data["df_house_mini"], test_size=0.4, verbose=False)
    report = analyzer.regress(
        models=[
            tm.ml.LinearR(
                type="l2",
                n_trials=1,
            ),
            tm.ml.CustomR(estimator=Ridge()),
        ],
        target="SalePrice",
        predictors=[
            "MSZoning",
            "ExterQual_binary",
            "LotArea",
            "OverallQual",
        ],
        feature_selectors=[tm.fs.KBestFSR("r_regression", 2)],
    )
    report._to_dict()


def test_regression_run_cv_fs(setup_data):
    """Tests regression mechanism for prediction with ml models"""
    analyzer = tm.Analyzer(setup_data["df_house_mini"], test_size=0.4, verbose=False)
    report = analyzer.regress(
        models=[
            tm.ml.LinearR(
                type="l2",
                n_trials=1,
            ),
            tm.ml.CustomR(estimator=Ridge()),
        ],
        target="SalePrice",
        predictors=[
            "MSZoning",
            "ExterQual_binary",
            "LotArea",
            "OverallQual",
        ],
        feature_selectors=[tm.fs.KBestFSR("r_regression", 2)],
        outer_cv=2,
    )
    report.cv_metrics(False)
    report.cv_metrics(True)
    report.feature_importance("LinearR(l2)")
    report.plot_obs_vs_pred("LinearR(l2)", "train")

    report._to_dict()


def test_classification_run_simple(setup_data):
    analyzer = tm.Analyzer(setup_data["df_house_mini"], test_size=0.4, verbose=False)
    report = analyzer.classify(
        models=[
            tm.ml.LinearC(
                type="l2",
                n_trials=1,
            ),
            tm.ml.CustomC(estimator=LogisticRegression(), name="customlogistic"),
        ],
        target="ExterQual_binary",
        predictors=[
            "MSZoning",
            "SalePrice",
            "LotArea",
            "OverallQual",
        ],
        feature_selectors=[tm.fs.KBestFSC("f_classif", 2)],
    )
    assert len(report.model("LinearC(l2)")._test_scorer._y_pred) == SAMPLE_SIZE * 0.4

    report.plot_confusion_matrix("LinearC(l2)", "train")
    report.plot_confusion_matrix("LinearC(l2)", "test")
    report.plot_roc_curve("LinearC(l2)", "train")
    report.plot_roc_curve("LinearC(l2)", "test")
    report.plot_roc_curves("train")
    report.plot_roc_curves("test")

    report._to_dict()
