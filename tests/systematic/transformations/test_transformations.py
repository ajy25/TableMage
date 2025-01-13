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


def test_scaling_simple(setup_data):
    df: pd.DataFrame = setup_data["df_house"].copy()

    # do not consider the test set
    analyzer = tm.Analyzer(df=df, test_size=0.0)

    # first, let's scale a variable with the minmax strategy
    analyzer.scale(include_vars=["SalePrice"], strategy="minmax")
    assert np.allclose(analyzer.df_all()["SalePrice"].min(), 0)
    assert np.allclose(analyzer.df_all()["SalePrice"].max(), 1)
    # save it to a checkpoint
    analyzer.save_data_checkpoint("minmaxed")

    # then, let's undo the scaling
    analyzer.load_data_checkpoint()
    assert np.allclose(analyzer.df_all()["SalePrice"].min(), df["SalePrice"].min())
    assert np.allclose(analyzer.df_all()["SalePrice"].max(), df["SalePrice"].max())

    # now, let's scale a variable with the standard strategy
    analyzer.scale(include_vars=["SalePrice"], strategy="standardize")
    assert np.allclose(analyzer.df_all()["SalePrice"].mean(), 0)
    assert np.allclose(analyzer.df_all()["SalePrice"].std(), 1, atol=1e-3)
    # save it to a checkpoint
    analyzer.save_data_checkpoint("standardized")

    # then, let's undo the scaling
    analyzer.load_data_checkpoint()
    assert np.allclose(analyzer.df_all()["SalePrice"].mean(), df["SalePrice"].mean())
    assert np.allclose(
        analyzer.df_all()["SalePrice"].std(), df["SalePrice"].std(), atol=1e-3
    )

    # now, let's scale a variable with the robust strategy
    analyzer.scale(include_vars=["SalePrice"], strategy="robust_standardize")
    assert np.allclose(
        analyzer.df_all()["SalePrice"].mean(),
        RobustScaler(unit_variance=True).fit_transform(df[["SalePrice"]]).mean(),
    )
    assert np.allclose(
        analyzer.df_all()["SalePrice"].std(),
        RobustScaler(unit_variance=True).fit_transform(df[["SalePrice"]]).std(),
        atol=1e-3,
    )

    # save it to a checkpoint
    analyzer.save_data_checkpoint("robust_standardized")

    # then, let's undo the scaling
    analyzer.load_data_checkpoint()

    assert np.allclose(analyzer.df_all()["SalePrice"].mean(), df["SalePrice"].mean())

    # now, let's scale a variable with the quantile strategy
    analyzer.scale(include_vars=["SalePrice"], strategy="normal_quantile")
    assert np.allclose(analyzer.df_all()["SalePrice"].mean(), 0, atol=1e-3)

    # load a checkpoint
    analyzer.load_data_checkpoint("minmaxed")

    # ensure that the scaling is correct
    assert np.allclose(analyzer.df_all()["SalePrice"].min(), 0)
    assert np.allclose(analyzer.df_all()["SalePrice"].max(), 1)

    # perform an additional normal scaling
    analyzer.scale(include_vars=["SalePrice"], strategy="normal_quantile")
    assert np.allclose(analyzer.df_all()["SalePrice"].mean(), 0, atol=1e-3)

    # obtain the scalers
    saleprice_scaler = analyzer.datahandler().scaler("SalePrice")

    # there should be two scalers
    assert len(saleprice_scaler) == 2

    # let's undo the transformation
    series_to_test = analyzer.df_all()["SalePrice"].copy()
    series_to_test = saleprice_scaler.inverse_transform(
        series_to_test.to_numpy().flatten()
    )
    assert np.allclose(series_to_test, df["SalePrice"])

    # test the remaining transformation strategies
    analyzer.load_data_checkpoint()

    analyzer.scale(include_vars=["SalePrice"], strategy="log")
    assert np.allclose(
        analyzer.df_all()["SalePrice"].mean(), np.log(df["SalePrice"]).mean(), atol=1e-3
    ), f"{analyzer.df_all()['SalePrice'].mean()}, {np.log(df['SalePrice']).mean()}"

    analyzer.load_data_checkpoint()
    analyzer.scale(include_vars=["SalePrice"], strategy="log1p")
    assert np.allclose(
        analyzer.df_all()["SalePrice"].mean(),
        np.log1p(df["SalePrice"]).mean(),
        atol=1e-3,
    )

    analyzer.load_data_checkpoint()
    analyzer.scale(include_vars=["SalePrice"], strategy="uniform_quantile")
    assert np.allclose(analyzer.df_all()["SalePrice"].mean(), 0.5, atol=1e-3)


def test_imputation_simple(setup_data):
    df: pd.DataFrame = setup_data["df_house"].copy()

    # let's only consider three numeric variables and two categorical variables
    df = df[["SalePrice", "LotFrontage", "LotArea", "GarageType", "GarageFinish"]]

    # randomly remove 10% of the data for each variable
    np.random.seed(42)
    for col in df.columns:
        df.loc[np.random.choice(df.index, int(len(df) * 0.1), replace=False), col] = (
            np.nan
        )

    analyzer = tm.Analyzer(df=df, test_size=0.0)
    analyzer.impute(
        include_vars=["SalePrice", "LotFrontage", "LotArea"], numeric_strategy="mean"
    )
    assert np.allclose(analyzer.df_all()["SalePrice"].mean(), df["SalePrice"].mean())

    analyzer.load_data_checkpoint()
    analyzer.impute(
        include_vars=["SalePrice", "LotFrontage", "LotArea"], numeric_strategy="5nn"
    )

    assert np.allclose(
        analyzer.df_all()["SalePrice"].to_numpy(),
        KNNImputer(n_neighbors=5, keep_empty_features=True).fit_transform(
            df[["SalePrice", "LotFrontage", "LotArea"]]
        )[:, 0],
    )

    analyzer.load_data_checkpoint()
    analyzer.impute(
        include_vars=["SalePrice", "LotFrontage", "LotArea"], numeric_strategy="10nn"
    )

    assert np.allclose(
        analyzer.df_all()["SalePrice"].to_numpy(),
        KNNImputer(n_neighbors=10, keep_empty_features=True).fit_transform(
            df[["SalePrice", "LotFrontage", "LotArea"]]
        )[:, 0],
    )

    analyzer.load_data_checkpoint()
    analyzer.impute(
        include_vars=["GarageType", "GarageFinish"], categorical_strategy="missing"
    )

    n_missing_garage_type = np.sum(df["GarageType"].isnull())
    n_missing_garage_finish = np.sum(df["GarageFinish"].isnull())

    assert "tm_missing" in analyzer.df_all()["GarageType"].unique()
    assert "tm_missing" in analyzer.df_all()["GarageFinish"].unique()

    assert (
        np.sum(analyzer.df_all()["GarageType"] == "tm_missing") == n_missing_garage_type
    )
    assert (
        np.sum(analyzer.df_all()["GarageFinish"] == "tm_missing")
        == n_missing_garage_finish
    )


def test_checkpoint_handling(setup_data):
    df: pd.DataFrame = setup_data["df_house"].copy()

    # let's only consider three numeric variables and two categorical variables
    df = df[["SalePrice", "LotFrontage", "LotArea", "GarageType", "GarageFinish"]]

    # randomly remove 10% of the data for each variable
    np.random.seed(42)
    for col in df.columns:
        df.loc[np.random.choice(df.index, int(len(df) * 0.1), replace=False), col] = (
            np.nan
        )

    analyzer = tm.Analyzer(df=df, test_size=0.0)
    analyzer.impute(
        include_vars=["SalePrice", "LotFrontage", "LotArea"], numeric_strategy="mean"
    )

    # save the data to a checkpoint
    analyzer.save_data_checkpoint("imputed")

    # load the data from the checkpoint
    analyzer.load_data_checkpoint("imputed")

    # ensure that the data is the same
    assert np.allclose(analyzer.df_all()["SalePrice"].mean(), df["SalePrice"].mean())

    # now, let's scale a variable with the minmax strategy
    analyzer.scale(include_vars=["SalePrice"], strategy="minmax")
    assert np.allclose(analyzer.df_all()["SalePrice"].min(), 0)
    assert np.allclose(analyzer.df_all()["SalePrice"].max(), 1)

    # save it to a checkpoint
    analyzer.save_data_checkpoint("imputed-then-minmaxed")

    # load the data from the checkpoint
    analyzer.load_data_checkpoint("imputed")

    # ensure that the data is the same
    assert np.allclose(analyzer.df_all()["SalePrice"].mean(), df["SalePrice"].mean())

    # load the data from the checkpoint
    analyzer.load_data_checkpoint("imputed-then-minmaxed")

    # ensure that the scaling is correct
    assert np.allclose(analyzer.df_all()["SalePrice"].min(), 0)
    assert np.allclose(analyzer.df_all()["SalePrice"].max(), 1)

    # now, let's scale a variable with the standard strategy
    analyzer.scale(include_vars=["SalePrice"], strategy="standardize")
    assert np.allclose(analyzer.df_all()["SalePrice"].mean(), 0)
    assert np.allclose(analyzer.df_all()["SalePrice"].std(), 1, atol=1e-3)

    analyzer.save_data_checkpoint("imputed-then-minmaxed-then-standardized")

    # load the data from the checkpoint
    analyzer.load_data_checkpoint("imputed")
    assert len(analyzer.datahandler()._preprocess_step_tracer._steps) == 1

    # ensure that the data is the same
    assert np.allclose(analyzer.df_all()["SalePrice"].mean(), df["SalePrice"].mean())

    analyzer.load_data_checkpoint()

    assert (
        len(analyzer.datahandler()._preprocess_step_tracer._steps) == 0
    ), "There should be no preprocessing steps"

    analyzer.load_data_checkpoint("imputed-then-minmaxed-then-standardized")
    assert len(analyzer.datahandler()._preprocess_step_tracer._steps) == 3
