import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
import pathlib
import sys
import matplotlib.pyplot as plt

parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(parent_dir))


import tablemage as tm


@pytest.fixture
def setup_data() -> dict:
    df_house = pd.read_csv(
        parent_dir / "demo" / "regression" / "house_price_data" / "data.csv"
    )
    df_house_train, df_house_test = train_test_split(
        df_house, test_size=0.2, random_state=42
    )
    return {
        "df_house": df_house,
        "df_house_train": df_house_train,
        "df_house_test": df_house_test,
    }


def test_simple_target_scaling_ols(setup_data):
    df_house_train = setup_data["df_house_train"].copy()
    df_house_test = setup_data["df_house_test"].copy()

    analyzer = tm.Analyzer(df=df_house_train, df_test=df_house_test)

    # first, fit an ols model and get the rmse
    report = analyzer.ols("SalePrice", predictors=["GrLivArea", "YearBuilt"])
    rmse = report.metrics("test").loc["rmse"]

    # let's scale the target variable
    analyzer.scale("SalePrice", strategy="standardize")
    report = analyzer.ols("SalePrice", predictors=["GrLivArea", "YearBuilt"])
    rmse_scaled = report.metrics("test").loc["rmse"]

    # scaling should be automatically inversed at last step
    assert np.allclose(rmse_scaled, rmse)

    # let's scale the target variable with minmax
    analyzer.scale("SalePrice", strategy="minmax")
    report = analyzer.ols("SalePrice", predictors=["GrLivArea", "YearBuilt"])
    rmse_scaled = report.metrics("test").loc["rmse"]

    # scaling should be automatically inversed at last step
    assert np.allclose(rmse_scaled, rmse)


def test_simple_target_scaling_ml(setup_data):
    df_house_train = setup_data["df_house_train"].copy()
    df_house_test = setup_data["df_house_test"].copy()

    analyzer = tm.Analyzer(df=df_house_train, df_test=df_house_test)

    # first, fit an ml model and get the rmse
    report = analyzer.regress(
        target="SalePrice",
        predictors=["GrLivArea", "YearBuilt"],
        models=[tm.ml.TreesR("decision_tree")],
    )
    rmse = report.metrics("test").loc["rmse"]

    # let's scale the target variable
    analyzer.scale("SalePrice", strategy="standardize")
    report = analyzer.regress(
        target="SalePrice",
        predictors=["GrLivArea", "YearBuilt"],
        models=[tm.ml.TreesR("decision_tree")],
    )
    rmse_scaled = report.metrics("test").loc["rmse"]

    # scaling should be automatically inversed at last step
    assert np.allclose(rmse_scaled, rmse)

    # let's scale the target variable with minmax
    analyzer.scale("SalePrice", strategy="minmax")
    report = analyzer.regress(
        target="SalePrice",
        predictors=["GrLivArea", "YearBuilt"],
        models=[tm.ml.TreesR("decision_tree")],
    )
    rmse_scaled = report.metrics("test").loc["rmse"]

    # scaling should be automatically inversed at last step
    assert np.allclose(rmse_scaled, rmse)
