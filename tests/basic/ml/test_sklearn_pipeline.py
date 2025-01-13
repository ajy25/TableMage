import pathlib
import sys
import pandas as pd
import pytest
import numpy as np
from sklearn.model_selection import train_test_split


parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(parent_dir))


import tablemage as tm

SAMPLE_SIZE = 150


@pytest.fixture
def setup_data():
    df_house = pd.read_csv(
        parent_dir / "demo" / "regression" / "house_price_data" / "data.csv"
    )
    df_house["ExterQual_binary"] = (df_house["ExterQual"] == "TA").astype(bool)

    df_house = df_house[
        [
            "MSZoning",
            "ExterQual_binary",
            "YearBuilt",
            "LotArea",
            "OverallQual",
            "SalePrice",
            "TotalBsmtSF",
            "1stFlrSF",
            "2ndFlrSF",
        ]
    ]

    df_house_mini = df_house.sample(SAMPLE_SIZE, random_state=42)

    df_house_mini_train, df_house_mini_test = train_test_split(
        df_house_mini, test_size=0.2, random_state=42
    )

    return {
        "df_house_mini_train": df_house_mini_train,
        "df_house_mini_test": df_house_mini_test,
    }


def test_pipeline_generation_regression(setup_data):
    """Tests pipeline generation for prediction with regression ml models"""

    # STEP 1
    # test basic functionality of the pipeline
    train_data = setup_data["df_house_mini_train"]
    test_data = setup_data["df_house_mini_test"]

    analyzer = tm.Analyzer(train_data, test_data, verbose=True)

    print(analyzer.datahandler().df_test().dtypes)
    print(analyzer.datahandler().df_test().dtypes)

    report = analyzer.regress(
        models=[
            tm.ml.LinearR(
                type="l2",
                n_trials=1,
            ),
        ],
        target="SalePrice",
        predictors=[
            "MSZoning",
            "ExterQual_binary",
            "LotArea",
            "OverallQual",
        ],
    )
    pipeline = report.model("LinearR(l2)").sklearn_pipeline()

    output = pipeline.predict(
        test_data[["MSZoning", "ExterQual_binary", "LotArea", "OverallQual"]]
    )

    assert np.allclose(
        output, report.model("LinearR(l2)")._test_scorer._y_pred, atol=1e-5
    )

    # STEP 2
    # test the pipeline with a feature selector
    report = analyzer.regress(
        models=[
            tm.ml.LinearR(
                type="l2",
                n_trials=1,
            ),
        ],
        target="SalePrice",
        predictors=[
            "MSZoning",
            "ExterQual_binary",
            "LotArea",
            "OverallQual",
        ],
        feature_selectors=[tm.fs.KBestFSR("f_regression", 2)],
    )
    pipeline = report.model("LinearR(l2)").sklearn_pipeline()

    output = pipeline.predict(
        test_data[["MSZoning", "ExterQual_binary", "LotArea", "OverallQual"]]
    )

    assert np.allclose(
        output, report.model("LinearR(l2)")._test_scorer._y_pred, atol=1e-5
    )

    # STEP 3
    # test the pipeline with a feature selector for specific model
    report = analyzer.regress(
        models=[
            tm.ml.LinearR(
                type="l2",
                n_trials=1,
                feature_selectors=[tm.fs.KBestFSR("f_regression", 3)],
            ),
        ],
        target="SalePrice",
        predictors=[
            "MSZoning",
            "ExterQual_binary",
            "LotArea",
            "OverallQual",
        ],
    )
    pipeline = report.model("LinearR(l2)").sklearn_pipeline()

    output = pipeline.predict(
        test_data[["MSZoning", "ExterQual_binary", "LotArea", "OverallQual"]]
    )

    assert np.allclose(
        output, report.model("LinearR(l2)")._test_scorer._y_pred, atol=1e-5
    )

    # STEP 4
    # test the pipeline with feature scaling
    analyzer.load_data_checkpoint()
    analyzer.scale(include_vars=["LotArea", "OverallQual"], strategy="standardize")

    report = analyzer.regress(
        models=[
            tm.ml.LinearR(
                type="l2",
                n_trials=1,
            ),
        ],
        target="SalePrice",
        predictors=[
            "MSZoning",
            "ExterQual_binary",
            "LotArea",
            "OverallQual",
        ],
    )
    report._to_dict()

    pipeline = report.model("LinearR(l2)").sklearn_pipeline()

    output = pipeline.predict(
        test_data[["MSZoning", "ExterQual_binary", "LotArea", "OverallQual"]]
    )

    assert np.allclose(
        output, report.model("LinearR(l2)")._test_scorer._y_pred, atol=1e-5
    )

    # STEP 5
    # test the pipeline with feature engineering
    analyzer.load_data_checkpoint()
    analyzer.engineer_numeric_var("TotalSF", "TotalBsmtSF + 1stFlrSF + 2ndFlrSF")

    report = analyzer.regress(
        models=[
            tm.ml.LinearR(
                type="l2",
                n_trials=1,
            ),
        ],
        target="SalePrice",
        predictors=[
            "MSZoning",
            "ExterQual_binary",
            "LotArea",
            "TotalSF",
        ],
    )
    pipeline = report.model("LinearR(l2)").sklearn_pipeline()

    output = pipeline.predict(
        test_data[
            [
                "MSZoning",
                "ExterQual_binary",
                "LotArea",
                "TotalBsmtSF",
                "1stFlrSF",
                "2ndFlrSF",
            ]
        ]
    )

    assert np.allclose(
        output, report.model("LinearR(l2)")._test_scorer._y_pred, atol=1e-5
    )

    # STEP 6
    # test the pipeline with categorical feature engineering
    analyzer.load_data_checkpoint()
    analyzer.engineer_categorical_var(
        name="1stFlrSF_quartiles",
        numeric_var="1stFlrSF",
        level_names=["Q1", "Q2", "Q3", "Q4"],
        thresholds=[882, 1087, 1391],
        leq=True,
    )

    report = analyzer.regress(
        models=[
            tm.ml.LinearR(
                type="l2",
                n_trials=1,
            ),
        ],
        target="SalePrice",
        predictors=[
            "MSZoning",
            "ExterQual_binary",
            "LotArea",
            "1stFlrSF_quartiles",
        ],
    )
    pipeline = report.model("LinearR(l2)").sklearn_pipeline()

    output = pipeline.predict(
        test_data[
            [
                "MSZoning",
                "ExterQual_binary",
                "LotArea",
                "TotalBsmtSF",
                "1stFlrSF",
                "2ndFlrSF",
            ]
        ]
    )

    assert np.allclose(
        output, report.model("LinearR(l2)")._test_scorer._y_pred, atol=1e-5
    )


def test_pipeline_generation_classification(setup_data):
    """Tests pipeline generation for prediction with classification ml models"""

    # STEP 1
    # test basic functionality of the pipeline
    train_data = setup_data["df_house_mini_train"]
    test_data = setup_data["df_house_mini_test"]

    analyzer = tm.Analyzer(train_data, test_data, verbose=False)

    report = analyzer.classify(
        models=[
            tm.ml.LinearC(
                type="l2",
                n_trials=1,
            ),
        ],
        target="ExterQual_binary",
        predictors=[
            "MSZoning",
            "LotArea",
            "OverallQual",
            "YearBuilt",
        ],
    )
    pipeline = report.model("LinearC(l2)").sklearn_pipeline()

    output = pipeline.predict_proba(
        test_data[["MSZoning", "LotArea", "OverallQual", "YearBuilt"]]
    )[:, 1]

    assert np.allclose(
        output, report.model("LinearC(l2)")._test_scorer._y_pred_score, atol=1e-5
    )

    # STEP 2
    # test the pipeline with a feature selector
    report = analyzer.classify(
        models=[
            tm.ml.LinearC(
                type="l2",
                n_trials=1,
            ),
        ],
        target="ExterQual_binary",
        predictors=[
            "MSZoning",
            "LotArea",
            "OverallQual",
            "YearBuilt",
        ],
        feature_selectors=[tm.fs.KBestFSC("f_classif", 2)],
    )
    pipeline = report.model("LinearC(l2)").sklearn_pipeline()

    output = pipeline.predict_proba(
        test_data[["MSZoning", "LotArea", "OverallQual", "YearBuilt"]]
    )[:, 1]

    assert np.allclose(
        output, report.model("LinearC(l2)")._test_scorer._y_pred_score, atol=1e-5
    )

    # STEP 3
    # test the pipeline with a feature selector for specific model
    report = analyzer.classify(
        models=[
            tm.ml.LinearC(
                type="l2",
                n_trials=1,
                feature_selectors=[tm.fs.KBestFSC("f_classif", 3)],
            ),
        ],
        target="ExterQual_binary",
        predictors=[
            "MSZoning",
            "LotArea",
            "OverallQual",
            "YearBuilt",
        ],
    )
    pipeline = report.model("LinearC(l2)").sklearn_pipeline()

    output = pipeline.predict_proba(
        test_data[["MSZoning", "LotArea", "OverallQual", "YearBuilt"]]
    )[:, 1]

    assert np.allclose(
        output, report.model("LinearC(l2)")._test_scorer._y_pred_score, atol=1e-5
    )

    # STEP 4
    # test the pipeline with feature scaling
    analyzer.load_data_checkpoint()
    analyzer.scale(include_vars=["LotArea", "OverallQual"], strategy="standardize")
    report = analyzer.classify(
        models=[
            tm.ml.LinearC(
                type="l2",
                n_trials=1,
            ),
        ],
        target="ExterQual_binary",
        predictors=[
            "MSZoning",
            "LotArea",
            "OverallQual",
            "YearBuilt",
        ],
    )
    pipeline = report.model("LinearC(l2)").sklearn_pipeline()

    output = pipeline.predict_proba(
        test_data[["MSZoning", "LotArea", "OverallQual", "YearBuilt"]]
    )[:, 1]

    assert np.allclose(
        output, report.model("LinearC(l2)")._test_scorer._y_pred_score, atol=1e-5
    )

    # STEP 5
    # test the pipeline with feature engineering
    analyzer.load_data_checkpoint()
    analyzer.engineer_numeric_var("TotalSF", "TotalBsmtSF + 1stFlrSF + 2ndFlrSF")

    report = analyzer.classify(
        models=[
            tm.ml.LinearC(
                type="l2",
                n_trials=1,
            ),
        ],
        target="ExterQual_binary",
        predictors=[
            "MSZoning",
            "LotArea",
            "OverallQual",
            "TotalSF",
        ],
    )

    pipeline = report.model("LinearC(l2)").sklearn_pipeline()

    output = pipeline.predict_proba(
        test_data[
            [
                "MSZoning",
                "LotArea",
                "OverallQual",
                "TotalBsmtSF",
                "1stFlrSF",
                "2ndFlrSF",
            ]
        ]
    )[:, 1]

    assert np.allclose(
        output, report.model("LinearC(l2)")._test_scorer._y_pred_score, atol=1e-5
    )

    # STEP 6
    # test the pipeline with categorical feature engineering
    analyzer.load_data_checkpoint()
    analyzer.engineer_categorical_var(
        name="1stFlrSF_quartiles",
        numeric_var="1stFlrSF",
        level_names=["Q1", "Q2", "Q3", "Q4"],
        thresholds=[882, 1087, 1391],
        leq=True,
    )

    report = analyzer.classify(
        models=[
            tm.ml.LinearC(
                type="l2",
                n_trials=1,
            ),
        ],
        target="ExterQual_binary",
        predictors=[
            "MSZoning",
            "LotArea",
            "OverallQual",
            "1stFlrSF_quartiles",
        ],
    )
    report._to_dict()

    pipeline = report.model("LinearC(l2)").sklearn_pipeline()

    output = pipeline.predict_proba(
        test_data[
            [
                "MSZoning",
                "LotArea",
                "OverallQual",
                "TotalBsmtSF",
                "1stFlrSF",
                "2ndFlrSF",
            ]
        ]
    )[:, 1]

    assert np.allclose(
        output, report.model("LinearC(l2)")._test_scorer._y_pred_score, atol=1e-5
    )


def test_pipeline_inverse_scaling_regression(setup_data):
    train_data = setup_data["df_house_mini_train"]
    test_data = setup_data["df_house_mini_test"]

    analyzer = tm.Analyzer(train_data, test_data, verbose=False)

    analyzer.scale(strategy="standardize")

    report = analyzer.regress(
        models=[
            tm.ml.LinearR(
                type="l2",
                n_trials=1,
            ),
        ],
        target="SalePrice",
        predictors=[
            "MSZoning",
            "ExterQual_binary",
            "LotArea",
            "OverallQual",
        ],
    )
    pipeline = report.model("LinearR(l2)").sklearn_pipeline()

    output = pipeline.predict(test_data)

    assert np.allclose(
        output, report.model("LinearR(l2)")._test_scorer._y_pred, atol=1e-5
    )
