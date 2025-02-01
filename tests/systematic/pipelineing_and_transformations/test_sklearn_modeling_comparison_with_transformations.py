import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
import pathlib
import sys


parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(parent_dir))


import tablemage as tm


@pytest.fixture
def setup_data() -> dict:
    df_house = pd.read_csv(
        parent_dir / "demo" / "regression" / "house_price_data" / "data.csv"
    )
    df_house["SalePrice"] = df_house["SalePrice"].astype(float) / 1000
    df_house_train, df_house_test = train_test_split(
        df_house, test_size=0.2, random_state=42
    )
    # force columns to either be numeric or str
    for col in df_house.columns:
        if df_house[col].dtype == "O":
            df_house[col] = df_house[col].astype(str)
        else:
            df_house[col] = df_house[col].astype(float)
    return {
        "df_house": df_house,
        "df_house_train": df_house_train,
        "df_house_test": df_house_test,
    }


def test_model_pipeline_with_transformations_against_sklearn(setup_data):
    df_train: pd.DataFrame = setup_data["df_house_train"].copy()
    df_test: pd.DataFrame = setup_data["df_house_test"].copy()

    # initialize the analyzer
    analyzer = tm.Analyzer(df=df_train, df_test=df_test)

    # steps: 1. select subset of variables, 2. impute, 3. minmax, 4. onehot, 5. standardize
    vars = [
        "SalePrice",
        "LotFrontage",
        "LotArea",
        "OverallQual",
        "OverallCond",
        "MSZoning",
    ]
    X_vars = vars.copy()
    X_vars.remove("SalePrice")
    analyzer.select_vars(vars)
    analyzer.impute(
        categorical_strategy="most_frequent",
        numeric_strategy="5nn",
        exclude_vars=["SalePrice"],
    ).scale(
        strategy="minmax",
        exclude_vars=["SalePrice"],
    ).onehot(
        dropfirst=False, exclude_vars=["SalePrice"]
    ).scale(
        strategy="standardize",
    )
    assert analyzer.df_train().mean().mean() < 1e-15

    y_test_col = analyzer.df_test()["SalePrice"].copy()
    y_scaler = analyzer._datahandler.scaler("SalePrice")
    y_test_col = y_scaler.inverse_transform(y_test_col)
    assert np.allclose(y_test_col, df_test["SalePrice"])

    # Next, train ML models
    ols_model = LinearRegression()
    ridge_model = Ridge(alpha=0.1, random_state=42)
    lasso_model = Lasso(alpha=0.1, random_state=42)

    # Try training custom versus default ols
    ml_report_temp = analyzer.regress(
        models=[
            tm.ml.LinearR("ols", name="ols_model"),
            tm.ml.CustomR(estimator=ols_model, name="ols_model_custom"),
        ],
        target="SalePrice",
    )
    assert np.allclose(
        ml_report_temp.model("ols_model")._test_scorer._y_pred,
        ml_report_temp.model("ols_model_custom")._test_scorer._y_pred,
    )
    tm_ols_pipeline = ml_report_temp.model("ols_model").sklearn_pipeline()
    tm_ols_pipeline_custom = ml_report_temp.model("ols_model_custom").sklearn_pipeline()
    assert np.allclose(
        tm_ols_pipeline.predict(df_test[X_vars]),
        tm_ols_pipeline_custom.predict(df_test[X_vars]),
    )

    ml_report = analyzer.regress(
        models=[
            tm.ml.LinearR("ols", name="ols_model"),
            tm.ml.CustomR(estimator=ols_model, name="ols_model_custom"),
            tm.ml.CustomR(estimator=ridge_model, name="ridge_model"),
            tm.ml.CustomR(estimator=lasso_model, name="lasso_model"),
        ],
        target="SalePrice",
    )
    tm_ols_pipeline = ml_report.model("ols_model").sklearn_pipeline()
    tm_ols_pipeline_custom = ml_report.model("ols_model_custom").sklearn_pipeline()
    tm_ridge_pipeline = ml_report.model("ridge_model").sklearn_pipeline()
    tm_lasso_pipeline = ml_report.model("lasso_model").sklearn_pipeline()

    assert np.allclose(
        ml_report.model("ols_model")._test_scorer._y_pred,
        ml_report.model("ols_model_custom")._test_scorer._y_pred,
    )
    assert np.allclose(
        ml_report.model("ols_model")._test_scorer._y_pred,
        tm_ols_pipeline_custom.predict(df_test[X_vars]),
    )
    assert np.allclose(
        ml_report.model("ridge_model")._test_scorer._y_pred,
        tm_ridge_pipeline.predict(df_test[X_vars]),
    )
    assert np.allclose(
        ml_report.model("lasso_model")._test_scorer._y_pred,
        tm_lasso_pipeline.predict(df_test[X_vars]),
    )

    # Define numeric and categorical columns
    numeric_cols = (
        df_train.select_dtypes(include=["number"]).columns.intersection(vars).to_list()
    )
    numeric_cols.remove("SalePrice")
    categorical_cols = df_train.select_dtypes(exclude=["number"]).columns.intersection(
        vars
    )

    # Define transformations for numeric columns
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", KNNImputer(n_neighbors=5)),
            ("minmax", MinMaxScaler()),
            ("scaler", StandardScaler()),
        ]
    )

    # Leave categorical columns as-is
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, drop="if_binary"
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer_ols = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, drop="first"
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    # Combine transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),  # Apply numeric transformations
            (
                "cat",
                categorical_transformer,
                categorical_cols,
            ),  # Leave categorical unchanged
        ]
    )
    preprocessor_ols = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),  # Apply numeric transformations
            (
                "cat",
                categorical_transformer_ols,
                categorical_cols,
            ),  # Leave categorical unchanged
        ]
    )

    # Create the pipeline
    sklearn_ols_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor_ols),
            ("regressor", ols_model),
        ]
    )
    sklearn_ols_pipeline = TransformedTargetRegressor(
        regressor=sklearn_ols_pipeline,
        transformer=StandardScaler(),  # Scales and unscales y
    )

    sklearn_ridge_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", ridge_model),
        ]
    )
    sklearn_ridge_pipeline = TransformedTargetRegressor(
        regressor=sklearn_ridge_pipeline,
        transformer=StandardScaler(),  # Scales and unscales y
    )

    sklearn_lasso_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", lasso_model),
        ]
    )
    sklearn_lasso_pipeline = TransformedTargetRegressor(
        regressor=sklearn_lasso_pipeline,
        transformer=StandardScaler(),  # Scales and unscales y
    )

    # Fit the pipeline
    sklearn_ols_pipeline.fit(df_train[X_vars], df_train["SalePrice"])
    sklearn_ridge_pipeline.fit(df_train[X_vars], df_train["SalePrice"])
    sklearn_lasso_pipeline.fit(df_train[X_vars], df_train["SalePrice"])

    # Compare the pipelines
    test_y_pred_tm_ridge = tm_ridge_pipeline.predict(df_test[X_vars])
    test_y_pred_tm_lasso = tm_lasso_pipeline.predict(df_test[X_vars])

    test_y_pred_sklearn_ridge = sklearn_ridge_pipeline.predict(df_test[X_vars])
    test_y_pred_sklearn_lasso = sklearn_lasso_pipeline.predict(df_test[X_vars])

    assert np.allclose(test_y_pred_tm_ridge, test_y_pred_sklearn_ridge)
    assert np.allclose(test_y_pred_tm_lasso, test_y_pred_sklearn_lasso, atol=1e-2)


def test_pipeline_generation_all_transformations(setup_data):
    df_train: pd.DataFrame = setup_data["df_house_train"].copy()
    df_test: pd.DataFrame = setup_data["df_house_test"].copy()

    # initialize the analyzer
    analyzer = tm.Analyzer(df=df_train, df_test=df_test)
    vars = [
        "SalePrice",
        "LotFrontage",
        "LotArea",
        "OverallQual",
        "OverallCond",
        "MSZoning",
        "HouseStyle",
        "1stFlrSF",
        "2ndFlrSF",
    ]
    # artificially introduce 10% missing values in SalePrice
    df_train.loc[df_train.sample(frac=0.1).index, "SalePrice"] = np.nan
    df_test.loc[df_test.sample(frac=0.1).index, "SalePrice"] = np.nan

    # step 0: drop rows with missing SalePrice
    analyzer.dropna(include_vars=["SalePrice"])

    # Step 1: select subset of variables
    analyzer.select_vars(vars)

    # Step 2: impute
    analyzer.impute(
        categorical_strategy="most_frequent",
        numeric_strategy="5nn",
        exclude_vars=["SalePrice"],
    )

    # Step 3: scale
    analyzer.scale(strategy="minmax")

    # Step 4: onehot
    analyzer.onehot(dropfirst=False, include_vars=["HouseStyle"])

    # Step 5: force binary
    analyzer.force_binary(var="MSZoning", pos_label="RL", rename=False)

    # Step 6: normalize
    analyzer.scale(strategy="normal_quantile", exclude_vars=["SalePrice"])

    # Step 7: enginner categorical
    analyzer.engineer_categorical_var(
        name="HighQual",
        numeric_var="OverallQual",
        level_names=["low", "high"],
        thresholds=[5],
        leq=True,
    )

    # Step 8: engineer numeric var
    analyzer.engineer_numeric_var(
        name="TotalSF",
        formula="1stFlrSF + 2ndFlrSF",
    )

    # Step 9: engineer interaction
    analyzer.engineer_numeric_var(
        name="TotalSF_OverallQual",
        formula="TotalSF * OverallQual",
    )

    report = analyzer.regress(
        models=[
            tm.ml.LinearR("ols", name="ols_model"),
            tm.ml.CustomR(estimator=LinearRegression(), name="ols_model_custom"),
        ],
        target="SalePrice",
        predictors=None,  # use all predictors
        feature_selectors=[
            tm.fs.KBestFSR("f_regression", k=5, name="kbest_f_regression"),
        ],
    )

    assert report.model("ols_model")._test_scorer._y_pred.shape[0] == df_test.shape[0]
    assert (
        report.model("ols_model_custom")._test_scorer._y_pred.shape[0]
        == df_test.shape[0]
    )

    X_vars = vars.copy()
    X_vars.remove("SalePrice")

    sklearn_ols_pipeline = report.model("ols_model").sklearn_pipeline()
    sklearn_ols_pipeline_custom = report.model("ols_model_custom").sklearn_pipeline()

    assert np.allclose(
        report.model("ols_model")._test_scorer._y_pred,
        sklearn_ols_pipeline.predict(df_test[X_vars]),
    )
    assert np.allclose(
        report.model("ols_model_custom")._test_scorer._y_pred,
        sklearn_ols_pipeline_custom.predict(df_test[X_vars]),
    )
