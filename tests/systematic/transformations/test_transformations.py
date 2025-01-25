import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    RobustScaler,
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
)
from sklearn.compose import ColumnTransformer
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


def test_scaling_simple(setup_data):
    atol = 1e-3
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
    assert np.allclose(analyzer.df_all()["SalePrice"].std(), 1, atol=atol)
    # save it to a checkpoint
    analyzer.save_data_checkpoint("standardized")

    # then, let's undo the scaling
    analyzer.load_data_checkpoint()
    assert np.allclose(analyzer.df_all()["SalePrice"].mean(), df["SalePrice"].mean())
    assert np.allclose(
        analyzer.df_all()["SalePrice"].std(), df["SalePrice"].std(), atol=atol
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
        atol=atol,
    )

    # save it to a checkpoint
    analyzer.save_data_checkpoint("robust_standardized")

    # then, let's undo the scaling
    analyzer.load_data_checkpoint()

    assert np.allclose(analyzer.df_all()["SalePrice"].mean(), df["SalePrice"].mean())

    # now, let's scale a variable with the quantile strategy
    analyzer.scale(include_vars=["SalePrice"], strategy="normal_quantile")
    assert np.allclose(analyzer.df_all()["SalePrice"].mean(), 0, atol=atol)

    # load a checkpoint
    analyzer.load_data_checkpoint("minmaxed")

    # ensure that the scaling is correct
    assert np.allclose(analyzer.df_all()["SalePrice"].min(), 0)
    assert np.allclose(analyzer.df_all()["SalePrice"].max(), 1)

    # perform an additional normal scaling
    analyzer.scale(include_vars=["SalePrice"], strategy="normal_quantile")
    assert np.allclose(analyzer.df_all()["SalePrice"].mean(), 0, atol=atol)

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
        analyzer.df_all()["SalePrice"].mean(), np.log(df["SalePrice"]).mean(), atol=atol
    ), f"{analyzer.df_all()['SalePrice'].mean()}, {np.log(df['SalePrice']).mean()}"

    analyzer.load_data_checkpoint()
    analyzer.scale(include_vars=["SalePrice"], strategy="log1p")
    assert np.allclose(
        analyzer.df_all()["SalePrice"].mean(),
        np.log1p(df["SalePrice"]).mean(),
        atol=atol,
    )

    analyzer.load_data_checkpoint()
    analyzer.scale(include_vars=["SalePrice"], strategy="uniform_quantile")
    assert np.allclose(analyzer.df_all()["SalePrice"].mean(), 0.5, atol=atol)


def test_transformations_against_sklearn(setup_data):
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
    analyzer.select_vars(vars)
    analyzer.impute(categorical_strategy="most_frequent", numeric_strategy="5nn").scale(
        strategy="minmax"
    ).onehot(dropfirst=False).scale(strategy="standardize")

    # sum along the columns, i.e. m x n -> m
    tm_test_nparray = analyzer.df_test().sum(axis=1)

    # Define numeric and categorical columns
    numeric_cols = df_train.select_dtypes(include=["number"]).columns.intersection(vars)
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
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
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

    # Create the pipeline
    sklearn_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
        ]
    )

    # Fit the pipeline on training data
    sklearn_pipeline.fit(df_train[vars])

    # Transform the test data
    transformed_test_data = sklearn_pipeline.transform(df_test[vars])

    # sum along the columns, i.e. m x n -> m
    sklearn_test_nparray = transformed_test_data.sum(axis=1)

    # Ensure that the two numpy arrays are the same
    assert np.allclose(tm_test_nparray, sklearn_test_nparray, atol=1e-6)


def test_checkpoint_handling(setup_data):
    atol = 1e-3
    df: pd.DataFrame = setup_data["df_house"].copy()

    # let's only consider three numeric variables and two categorical variables
    df = df[["SalePrice", "LotFrontage", "LotArea", "GarageType", "GarageFinish"]]

    # randomly remove 10% of the data for each variable
    np.random.seed(42)
    for col in df.columns:
        df.loc[
            np.random.choice(df.index, int(len(df) * 0.1), replace=False), col
        ] = np.nan

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
    assert np.allclose(analyzer.df_all()["SalePrice"].std(), 1, atol=atol)

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
