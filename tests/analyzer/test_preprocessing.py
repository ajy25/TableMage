import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pathlib
import sys

parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))


import tablemage as tm


@pytest.fixture
def setup_data():
    df_house = pd.read_csv(
        parent_dir / "demo" / "regression" / "house_price_data" / "data.csv"
    )
    df_house_train, df_house_test = train_test_split(
        df_house, test_size=0.2, random_state=42
    )

    return {
        "df_house_train": df_house_train,
        "df_house_test": df_house_test,
    }


def test_standard_scaling(setup_data):
    """Tests basic feature scaling functionality"""

    df_house_train = setup_data["df_house_train"]
    df_house_test = setup_data["df_house_test"]

    analyzer = tm.Analyzer(df_house_train, df_house_test, verbose=False)

    scaled_vars = ["GrLivArea", "YearBuilt", "OverallQual", "LotArea", "SalePrice"]

    analyzer.dropna(include_vars=scaled_vars)

    analyzer.scale(include_vars=scaled_vars, strategy="standardize")

    analyzer_working_df_train = analyzer.df_train()[scaled_vars]
    print(analyzer_working_df_train.shape)

    scaler = StandardScaler()

    assert np.allclose(
        analyzer_working_df_train.to_numpy(),
        scaler.fit_transform(df_house_train[scaled_vars].to_numpy()),
        atol=1e-5,
    )

    analyzer_working_df_test = analyzer.df_test()[scaled_vars]
    print(analyzer_working_df_test.shape)

    assert np.allclose(
        analyzer_working_df_test.to_numpy(),
        scaler.transform(df_house_test[scaled_vars].to_numpy()),
        atol=1e-5,
    )


def test_minmax_scaling(setup_data):
    """Tests basic feature scaling functionality"""

    df_house_train = setup_data["df_house_train"]
    df_house_test = setup_data["df_house_test"]

    analyzer = tm.Analyzer(df_house_train, df_house_test, verbose=False)

    scaled_vars = ["GrLivArea", "YearBuilt", "OverallQual", "LotArea", "SalePrice"]

    analyzer.scale(include_vars=scaled_vars, strategy="minmax")

    analyzer_working_df_train = analyzer.datahandler().df_train()[scaled_vars]

    scaler = MinMaxScaler()

    assert np.allclose(
        analyzer_working_df_train.to_numpy(),
        scaler.fit_transform(df_house_train[scaled_vars].to_numpy()),
        atol=1e-5,
    )

    analyzer_working_df_test = analyzer.datahandler().df_test()[scaled_vars]

    assert np.allclose(
        analyzer_working_df_test.to_numpy(),
        scaler.transform(df_house_test[scaled_vars].to_numpy()),
        atol=1e-5,
    )


def test_data_checkpoint_loading(setup_data):
    """Tests basic checkpointing functionality"""

    df_house_train = setup_data["df_house_train"]
    df_house_test = setup_data["df_house_test"]

    analyzer = tm.Analyzer(df_house_train, df_house_test, verbose=False)

    categorical_vars_with_missing = ["GarageFinish", "GarageCond"]

    numerical_vars = [
        "GrLivArea",
        "YearBuilt",
        "OverallQual",
        "LotFrontage",
        "LotArea",
        "SalePrice",
    ]

    X_vars = [
        "GrLivArea",
        "YearBuilt",
        "OverallQual",
        "LotFrontage",
        "LotArea",
    ]

    orig_missingness = analyzer.datahandler().df_train().isna().sum().sum()
    assert orig_missingness != 0

    orig_emitted_X_numpy = (
        analyzer.datahandler()
        .train_test_emitter(y_var="SalePrice", X_vars=X_vars)
        .emit_train_test_Xy()[0]
        .to_numpy()
    )

    orig_emitted_y_numpy = (
        analyzer.datahandler()
        .train_test_emitter(y_var="SalePrice", X_vars=X_vars)
        .emit_train_test_Xy()[1]
        .to_numpy()
    )

    # first, select some variables, drop missing in some
    analyzer.drop_highly_missing_vars()
    analyzer.select_vars(include_vars=categorical_vars_with_missing + numerical_vars)
    analyzer.dropna(include_vars=categorical_vars_with_missing)
    analyzer.save_data_checkpoint("cat_missing_dropped")

    analyzer_df = analyzer.datahandler().df_train().dropna()
    emitter = analyzer.datahandler().train_test_emitter(
        y_var="SalePrice", X_vars=X_vars
    )
    train_X, train_y, _, _ = emitter.emit_train_test_Xy()

    cat_missing_dropped_datahandler_X_numpy = analyzer_df[X_vars].to_numpy()
    cat_missing_dropped_dataemitter_X_numpy = train_X.to_numpy()

    cat_missing_dropped_datahandler_y_numpy = analyzer_df["SalePrice"].to_numpy()
    cat_missing_dropped_dataemitter_y_numpy = train_y.to_numpy()

    assert np.allclose(
        cat_missing_dropped_datahandler_X_numpy,
        cat_missing_dropped_dataemitter_X_numpy,
        atol=1e-5,
    )

    assert np.allclose(
        cat_missing_dropped_datahandler_y_numpy,
        cat_missing_dropped_dataemitter_y_numpy,
        atol=1e-5,
    )

    cat_missing_dropped_missingness = (
        analyzer.datahandler().df_train().isna().sum().sum()
    )
    assert cat_missing_dropped_missingness != 0
    assert cat_missing_dropped_missingness < orig_missingness

    # load above checkpoint and ensure all close
    analyzer.load_data_checkpoint("cat_missing_dropped")

    cat_missing_loaded_missingness = (
        analyzer.datahandler().df_train().isna().sum().sum()
    )

    cat_missing_dropped_dataemitter_loaded_X_numpy = (
        analyzer.datahandler()
        .train_test_emitter(y_var="SalePrice", X_vars=X_vars)
        .emit_train_test_Xy()[0]
        .to_numpy()
    )

    cat_missing_dropped_datahandler_loaded_X_numpy = (
        analyzer.datahandler().df_train().dropna()[X_vars].to_numpy()
    )

    assert cat_missing_loaded_missingness == cat_missing_dropped_missingness

    assert np.allclose(
        cat_missing_dropped_datahandler_loaded_X_numpy,
        cat_missing_dropped_datahandler_X_numpy,
        atol=1e-5,
    )

    assert np.allclose(
        cat_missing_dropped_dataemitter_loaded_X_numpy,
        cat_missing_dropped_dataemitter_X_numpy,
        atol=1e-5,
    )

    # revert to original data and ensure all close
    analyzer.load_data_checkpoint()

    orig_missingness_after_revert = analyzer.datahandler().df_train().isna().sum().sum()

    assert orig_missingness_after_revert == orig_missingness

    orig_reverted_X_numpy = (
        analyzer.datahandler()
        .train_test_emitter(y_var="SalePrice", X_vars=X_vars)
        .emit_train_test_Xy()[0]
        .to_numpy()
    )

    orig_reverted_y_numpy = (
        analyzer.datahandler()
        .train_test_emitter(y_var="SalePrice", X_vars=X_vars)
        .emit_train_test_Xy()[1]
        .to_numpy()
    )

    assert np.allclose(orig_reverted_X_numpy, orig_emitted_X_numpy, atol=1e-5)

    assert np.allclose(orig_reverted_y_numpy, orig_emitted_y_numpy, atol=1e-5)

    # load 'cat_missing_dropped' checkpoint and ensure all close
    analyzer.load_data_checkpoint("cat_missing_dropped")

    cat_missing_loaded_missingness = (
        analyzer.datahandler().df_train().isna().sum().sum()
    )

    cat_missing_dropped_dataemitter_loaded_X_numpy = (
        analyzer.datahandler()
        .train_test_emitter(y_var="SalePrice", X_vars=X_vars)
        .emit_train_test_Xy()[0]
        .to_numpy()
    )

    cat_missing_dropped_datahandler_loaded_X_numpy = (
        analyzer.datahandler().df_train().dropna()[X_vars].to_numpy()
    )

    assert cat_missing_loaded_missingness == cat_missing_dropped_missingness

    assert np.allclose(
        cat_missing_dropped_datahandler_loaded_X_numpy,
        cat_missing_dropped_datahandler_X_numpy,
        atol=1e-5,
    )

    assert np.allclose(
        cat_missing_dropped_dataemitter_loaded_X_numpy,
        cat_missing_dropped_dataemitter_X_numpy,
        atol=1e-5,
    )

    # scale and impute
    analyzer.scale().impute()
    assert analyzer.datahandler().df_train().isna().sum().sum() == 0

    # delete the checkpoint
    analyzer.remove_data_checkpoint("cat_missing_dropped")

    with pytest.raises(Exception):
        analyzer.load_data_checkpoint("cat_missing_dropped")
