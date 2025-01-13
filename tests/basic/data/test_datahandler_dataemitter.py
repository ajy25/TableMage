import pytest
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pathlib
import sys

parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(parent_dir))

from tablemage._src.data.datahandler import (
    DataEmitter,
    DataHandler,
    PreprocessStepTracer,
)


@pytest.fixture
def setup_data():
    df_simple = pd.DataFrame(
        {
            "binary_var": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
            "categorical_var": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
            "numeric_var": [5.2, 3.7, 8.1, 2.5, 6.9, 4.3, 7.8, 1.1, 3.4, 5.7],
        }
    )
    df_simple_train, df_simple_test = train_test_split(
        df_simple, test_size=0.2, random_state=42
    )
    iris = datasets.load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    target = pd.Series(iris.target, name="target")
    df_iris = pd.concat([data, target], axis=1)

    df_iris_train, df_iris_test = train_test_split(
        df_iris, test_size=0.2, random_state=42
    )

    df_house = pd.read_csv(
        parent_dir / "demo" / "regression" / "house_price_data" / "data.csv"
    )
    df_house_train, df_house_test = train_test_split(
        df_house, test_size=0.2, random_state=42
    )

    return {
        "df_simple_train": df_simple_train,
        "df_simple_test": df_simple_test,
        "df_iris_train": df_iris_train,
        "df_iris_test": df_iris_test,
        "df_house_train": df_house_train,
        "df_house_test": df_house_test,
    }


def test_basic_init(setup_data):
    """Test basic initialization DataEmitter functionality."""
    df_simple_train = setup_data["df_simple_train"]
    df_simple_test = setup_data["df_simple_test"]

    dh = DataHandler(df_simple_train, df_simple_test, verbose=False)
    de = DataEmitter(
        df_simple_train,
        df_simple_test,
        "binary_var",
        ["categorical_var", "numeric_var"],
        PreprocessStepTracer(),
    )
    dh_emitter = dh.train_test_emitter("binary_var", ["categorical_var", "numeric_var"])
    assert dh_emitter._working_df_test.shape == de._working_df_test.shape
    assert dh_emitter._working_df_train.shape == de._working_df_train.shape
    dh.drop_vars(["binary_var"])
    assert dh._working_df_train.shape != de._working_df_train.shape


def test_force_categorical(setup_data):
    """Test force categorical encoding of numeric or binary variables."""
    df_simple_train = setup_data["df_simple_train"]
    df_simple_test = setup_data["df_simple_test"]

    dh = DataHandler(df_simple_train, df_simple_test, verbose=False)
    de = DataEmitter(
        df_simple_train,
        df_simple_test,
        "binary_var",
        ["categorical_var", "numeric_var"],
        PreprocessStepTracer(),
    )
    dh.force_categorical(["binary_var"])
    dh_emitter = dh.train_test_emitter("binary_var", ["categorical_var", "numeric_var"])
    assert dh_emitter._working_df_train["binary_var"].dtype == "object"
    assert dh_emitter._working_df_test.shape == de._working_df_test.shape
    assert dh_emitter._working_df_train.shape == de._working_df_train.shape


# Continue converting the rest of the test methods in a similar manner...


def test_force_numeric(setup_data):
    """Test force numeric encoding of categorical variables."""
    df_simple_train = setup_data["df_simple_train"]
    df_simple_test = setup_data["df_simple_test"]

    dh = DataHandler(df_simple_train, df_simple_test, verbose=False)
    de = DataEmitter(
        df_simple_train,
        df_simple_test,
        "binary_var",
        ["categorical_var", "numeric_var"],
        PreprocessStepTracer(),
    )
    dh.force_numeric(["binary_var"])
    dh_emitter = dh.train_test_emitter("binary_var", ["categorical_var", "numeric_var"])
    assert dh_emitter._working_df_train["binary_var"].dtype == "float64"
    assert dh_emitter._working_df_test.shape == de._working_df_test.shape
    assert dh_emitter._working_df_train.shape == de._working_df_train.shape


def test_force_binary(setup_data):
    """Test force binary encoding of numeric or categorical variables."""
    df_simple_train = setup_data["df_simple_train"]
    df_simple_test = setup_data["df_simple_test"]

    dh = DataHandler(df_simple_train, df_simple_test, verbose=False)
    de = DataEmitter(
        df_simple_train,
        df_simple_test,
        "binary_var",
        ["categorical_var", "numeric_var"],
        PreprocessStepTracer(),
    )
    dh.force_binary(["numeric_var"], rename=True)
    dh_emitter = dh.train_test_emitter("binary_var", ["categorical_var", "numeric_var"])
    assert (
        dh_emitter._working_df_train["numeric_var"].dtype
        == de._working_df_train["numeric_var"].dtype
    )
    assert dh_emitter._working_df_test.shape == de._working_df_test.shape
    assert dh_emitter._working_df_train.shape == de._working_df_train.shape

    dh = DataHandler(df_simple_train, df_simple_test, verbose=False)
    de = DataEmitter(
        df_simple_train,
        df_simple_test,
        "binary_var",
        ["categorical_var", "numeric_var"],
        PreprocessStepTracer(),
    )
    de._force_binary(
        ["categorical_var"], pos_labels=["A"], ignore_multiclass=True, rename=True
    )
    dh.force_binary(
        ["categorical_var"], pos_labels=["A"], ignore_multiclass=True, rename=True
    )
    dh_emitter = dh.train_test_emitter(
        "binary_var", ["categorical_var::A", "numeric_var"]
    )
    assert (
        dh_emitter._working_df_train["categorical_var::A"].dtype
        == de._working_df_train["categorical_var::A"].dtype
    )
    assert dh_emitter._working_df_test.shape == de._working_df_test.shape
    assert dh_emitter._working_df_train.shape == de._working_df_train.shape

    dh.force_binary(
        ["numeric_var"], pos_labels=[5.2], ignore_multiclass=True, rename=True
    )
    dh_emitter = dh.train_test_emitter(
        "binary_var", ["categorical_var::A", "numeric_var::5.2"]
    )
    assert dh._working_df_train["numeric_var::5.2"].dtype == "int64"
    assert dh_emitter._working_df_train.shape == dh._working_df_train.shape


def test_onehot(setup_data):
    """Test onehot encoding of categorical variables."""
    df_simple_train = setup_data["df_simple_train"]
    df_simple_test = setup_data["df_simple_test"]

    dh = DataHandler(df_simple_train, df_simple_test, verbose=False)
    dh.onehot(["categorical_var"], dropfirst=False)
    assert all(
        [
            col in dh._working_df_train.columns
            for col in [
                "categorical_var::A",
                "categorical_var::B",
                "categorical_var::C",
            ]
        ]
    )
    assert all(
        [
            col in dh._working_df_test.columns
            for col in [
                "categorical_var::A",
                "categorical_var::B",
                "categorical_var::C",
            ]
        ]
    )

    dh_emitter = dh.train_test_emitter(
        "binary_var",
        [
            "categorical_var::A",
            "categorical_var::B",
            "categorical_var::C",
            "numeric_var",
        ],
    )
    assert all(
        [
            col in dh_emitter._working_df_test.columns
            for col in [
                "categorical_var::A",
                "categorical_var::B",
                "categorical_var::C",
            ]
        ]
    )
    assert all(
        [
            col in dh_emitter._working_df_train.columns
            for col in [
                "categorical_var::A",
                "categorical_var::B",
                "categorical_var::C",
            ]
        ]
    )
    assert dh_emitter._working_df_test.shape == dh._working_df_test.shape
    assert dh_emitter._working_df_train.shape == dh._working_df_train.shape

    dh = DataHandler(df_simple_train, df_simple_test, verbose=False)
    dh.onehot(["categorical_var", "binary_var"], dropfirst=True)
    assert all(
        [
            col in dh._working_df_train.columns
            for col in ["categorical_var::B", "categorical_var::C"]
        ]
    )
    assert all(
        [
            col in dh._working_df_test.columns
            for col in ["categorical_var::B", "categorical_var::C"]
        ]
    )
    assert "categorical_var::A" not in dh._working_df_train.columns

    dh_emitter = dh.train_test_emitter(
        "numeric_var",
        ["categorical_var::B", "categorical_var::C", "binary_var::1"],
    )
    assert all(
        [
            col in dh_emitter._working_df_test.columns
            for col in ["categorical_var::B", "categorical_var::C"]
        ]
    )
    assert all(
        [
            col in dh_emitter._working_df_train.columns
            for col in ["categorical_var::B", "categorical_var::C"]
        ]
    )
    assert dh_emitter._working_df_test.shape == dh._working_df_test.shape
    assert dh_emitter._working_df_train.shape == dh._working_df_train.shape


def test_onehot_keep_orig(setup_data):
    """Test onehot encoding of categorical variables with original columns."""
    df_simple_train = setup_data["df_simple_train"]
    df_simple_test = setup_data["df_simple_test"]

    dh = DataHandler(df_simple_train, df_simple_test, verbose=False)
    dh.onehot(["categorical_var"], dropfirst=False, keep_original=True)
    assert all(
        [
            col in dh._working_df_train.columns
            for col in [
                "categorical_var",
                "categorical_var::A",
                "categorical_var::B",
                "categorical_var::C",
            ]
        ]
    )
    assert all(
        [
            col in dh._working_df_test.columns
            for col in [
                "categorical_var",
                "categorical_var::A",
                "categorical_var::B",
                "categorical_var::C",
            ]
        ]
    )

    dh_emitter = dh.train_test_emitter(
        "binary_var",
        [
            "categorical_var",
            "categorical_var::A",
            "categorical_var::B",
            "categorical_var::C",
            "numeric_var",
        ],
    )
    assert all(
        [
            col in dh_emitter._working_df_test.columns
            for col in [
                "categorical_var",
                "categorical_var::A",
                "categorical_var::B",
                "categorical_var::C",
            ]
        ]
    )
    assert all(
        [
            col in dh_emitter._working_df_train.columns
            for col in [
                "categorical_var",
                "categorical_var::A",
                "categorical_var::B",
                "categorical_var::C",
            ]
        ]
    )
    assert dh_emitter._working_df_test.shape == dh._working_df_test.shape
    assert dh_emitter._working_df_train.shape == dh._working_df_train.shape

    dh = DataHandler(df_simple_train, df_simple_test, verbose=False)
    dh.onehot(["categorical_var", "binary_var"], dropfirst=True, keep_original=True)
    assert all(
        [
            col in dh._working_df_train.columns
            for col in [
                "categorical_var",
                "categorical_var::B",
                "categorical_var::C",
            ]
        ]
    )
    assert all(
        [
            col in dh._working_df_test.columns
            for col in [
                "categorical_var",
                "categorical_var::B",
                "categorical_var::C",
            ]
        ]
    )
    assert "categorical_var::A" not in dh._working_df_train.columns
    assert "categorical_var::A" not in dh._working_df_test.columns


def test_multiple(setup_data):
    """Test multiple preprocessing steps."""
    df_iris_train = setup_data["df_iris_train"]
    df_iris_test = setup_data["df_iris_test"]

    dh = DataHandler(df_iris_train, df_iris_test, verbose=False)
    print(dh)

    dh.force_categorical(["target"])
    dh.scale(["sepal_length_(cm)"], strategy="log")
    dh.drop_vars(["sepal_width_(cm)"])
    dh_emitter = dh.train_test_emitter(
        "target", ["sepal_length_(cm)", "petal_length_(cm)", "petal_width_(cm)"]
    )
    assert all(
        col1 == col2
        for col1, col2 in zip(
            dh_emitter._working_df_train.columns, dh._working_df_train.columns
        )
    )
    assert dh_emitter._working_df_train.shape == dh._working_df_train.shape
    assert all(
        idx1 == idx2
        for idx1, idx2 in zip(
            dh_emitter._working_df_train.index, dh._working_df_train.index
        )
    )


def test_scale(setup_data):
    """Test scaling of numeric variables."""
    df_iris_train = setup_data["df_iris_train"]
    df_iris_test = setup_data["df_iris_test"]

    dh = DataHandler(df_iris_train, df_iris_test, verbose=False)
    dh.scale(strategy="minmax")
    assert dh._working_df_train["sepal_length_(cm)"].min() == pytest.approx(0)
    assert dh._working_df_train["sepal_length_(cm)"].max() == pytest.approx(1)
    assert dh._working_df_train["petal_length_(cm)"].min() == pytest.approx(0)
    assert dh._working_df_train["petal_length_(cm)"].max() == pytest.approx(1)

    dh_emitter = dh.train_test_emitter(
        "target", ["sepal_length_(cm)", "petal_length_(cm)", "petal_width_(cm)"]
    )
    assert dh_emitter._working_df_train["sepal_length_(cm)"].min() == pytest.approx(0)
    assert dh_emitter._working_df_train["sepal_length_(cm)"].max() == pytest.approx(1)
    assert dh_emitter._working_df_train["petal_length_(cm)"].min() == pytest.approx(0)
    assert dh_emitter._working_df_train["petal_length_(cm)"].max() == pytest.approx(1)

    dh = DataHandler(df_iris_train, df_iris_test, verbose=False)
    dh.scale(strategy="standardize")
    assert dh._working_df_train["sepal_length_(cm)"].mean() == pytest.approx(
        0, abs=0.02
    )
    assert dh._working_df_train["sepal_length_(cm)"].std() == pytest.approx(1, abs=0.02)
    assert dh._working_df_train["petal_length_(cm)"].mean() == pytest.approx(
        0, abs=0.02
    )
    assert dh._working_df_train["petal_length_(cm)"].std() == pytest.approx(1, abs=0.02)

    dh_emitter = dh.train_test_emitter(
        "target", ["sepal_length_(cm)", "petal_length_(cm)", "petal_width_(cm)"]
    )
    assert dh_emitter._working_df_train["sepal_length_(cm)"].mean() == pytest.approx(
        0, abs=0.02
    )
    assert dh_emitter._working_df_train["sepal_length_(cm)"].std() == pytest.approx(
        1, abs=0.02
    )

    dh_emitters = dh.kfold_emitters(
        "target", ["sepal_length_(cm)", "petal_length_(cm)", "petal_width_(cm)"]
    )
    for emitter in dh_emitters:
        assert emitter._working_df_train["sepal_length_(cm)"].mean() == pytest.approx(
            0, abs=0.02
        )
        assert emitter._working_df_train["sepal_length_(cm)"].std() == pytest.approx(
            1, abs=0.1
        )


def test_drop_vars(setup_data):
    """Test dropping variables."""
    df_iris_train = setup_data["df_iris_train"]
    df_iris_test = setup_data["df_iris_test"]

    dh = DataHandler(df_iris_train, df_iris_test, verbose=False)
    dh.drop_vars(["sepal_length_(cm)"])
    assert "sepal_length_(cm)" not in dh._working_df_train.columns
    assert "sepal_length_(cm)" not in dh._working_df_test.columns

    dh_emitter = dh.train_test_emitter(
        "target", ["petal_length_(cm)", "petal_width_(cm)"]
    )
    assert "sepal_length_(cm)" not in dh_emitter._working_df_train.columns
    assert "sepal_length_(cm)" not in dh_emitter._working_df_test.columns


def test_impute(setup_data):
    """Test imputing missing values."""
    df_house_train = setup_data["df_house_train"]
    df_house_test = setup_data["df_house_test"]

    dh = DataHandler(df_house_train, df_house_test, verbose=False)
    dh.impute(numeric_strategy="mean", categorical_strategy="most_frequent")
    assert all(
        not dh._working_df_train[col].isnull().any()
        for col in dh._working_df_train.columns
    )
    assert all(
        not dh._working_df_test[col].isnull().any()
        for col in dh._working_df_test.columns
    )

    dh_emitter = dh.train_test_emitter(
        "SalePrice", ["LotFrontage", "LotArea", "OverallQual", "OverallCond"]
    )
    assert all(
        not dh_emitter._working_df_train[col].isnull().any()
        for col in dh_emitter._working_df_train.columns
    )
    assert all(
        not dh_emitter._working_df_test[col].isnull().any()
        for col in dh_emitter._working_df_test.columns
    )

    dh_emitters = dh.kfold_emitters(
        "SalePrice", ["LotFrontage", "LotArea", "OverallQual", "OverallCond"]
    )
    for emitter in dh_emitters:
        assert all(
            not emitter._working_df_train[col].isnull().any()
            for col in emitter._working_df_train.columns
        )
        assert all(
            not emitter._working_df_test[col].isnull().any()
            for col in emitter._working_df_test.columns
        )

    assert df_house_train["LotFrontage"].isnull().any()
    assert df_house_train["MasVnrArea"].isnull().any()
    dh = DataHandler(df_house_train, df_house_test, verbose=False)
    dh.impute(
        include_vars=["LotFrontage", "MasVnrArea", "BsmtFinSF1"],
        exclude_vars=["LotFrontage"],
    )
    assert dh._working_df_train["LotFrontage"].isnull().any()
    assert not dh._working_df_train["MasVnrArea"].isnull().any()
    assert df_house_train["MasVnrArea"].isnull().any()


def test_impute_agreement(setup_data):
    """Tests that imputation methods agree with the DataEmitter."""
    df_house_train = setup_data["df_house_train"]
    df_house_test = setup_data["df_house_test"]

    dh = DataHandler(df_house_train, df_house_test, verbose=False)

    dh.impute(numeric_strategy="mean", categorical_strategy="most_frequent")

    dh_emitter = dh.train_test_emitter(
        "SalePrice", ["LotFrontage", "LotArea", "OverallQual", "OverallCond"]
    )

    assert all(
        not dh_emitter._working_df_train[col].isnull().any()
        for col in dh_emitter._working_df_train.columns
    )
    assert all(
        not dh_emitter._working_df_test[col].isnull().any()
        for col in dh_emitter._working_df_test.columns
    )

    assert dh_emitter._working_df_train.equals(dh._working_df_train)

    dh.load_data_checkpoint()

    dh.impute(numeric_strategy="5nn", categorical_strategy="most_frequent")

    dh_emitter = dh.train_test_emitter(
        "SalePrice", ["LotFrontage", "LotArea", "OverallQual", "OverallCond"]
    )

    assert all(
        not dh_emitter._working_df_train[col].isnull().any()
        for col in dh_emitter._working_df_train.columns
    )
    assert all(
        not dh_emitter._working_df_test[col].isnull().any()
        for col in dh_emitter._working_df_test.columns
    )

    assert dh_emitter._working_df_train.equals(dh._working_df_train)


def test_kfold_basic_init(setup_data):
    """Test kfold cross validation basic functionality."""
    df_iris_train = setup_data["df_iris_train"]
    df_iris_test = setup_data["df_iris_test"]

    dh = DataHandler(df_iris_train, df_iris_test, verbose=False)
    dh.force_categorical(["target"])
    dh.scale(["sepal_length_(cm)"], strategy="log")
    dh.drop_vars(["sepal_width_(cm)"])
    emitters = dh.kfold_emitters(
        "target", ["sepal_length_(cm)", "petal_length_(cm)", "petal_width_(cm)"]
    )
    idxs = []
    for emitter in emitters:
        for col1, col2 in zip(
            emitter._working_df_train.columns, dh._working_df_train.columns
        ):
            assert col1 == col2
        idxs.append(emitter._working_df_train.index.to_list())
    idxs = np.concatenate(idxs)
    for idx in idxs:
        assert idx in dh._working_df_train.index

    dh = DataHandler(df_iris_train, df_iris_test, verbose=False)
    dh.force_categorical(["target"])
    dh.onehot(["target"])
    dh.scale(["sepal_length_(cm)"], strategy="minmax")
    dh.drop_vars(["sepal_width_(cm)"])
    emitters = dh.kfold_emitters(
        "sepal_length_(cm)",
        ["target::1", "target::2", "petal_length_(cm)", "petal_width_(cm)"],
        n_folds=5,
        shuffle=True,
        random_state=42,
    )
    idxs = []
    for emitter in emitters:
        for col1, col2 in zip(
            emitter._working_df_train.columns, dh._working_df_train.columns
        ):
            assert col1 == col2
    idxs.append(emitter._working_df_train.index.to_list())
    idxs = np.concatenate(idxs)
    for idx in idxs:
        assert idx in dh._working_df_train.index
    assert emitters[0].y_scaler().scalers[0].min == 4.3
    assert emitters[2].y_scaler().scalers[0].min != 4.3


def test_emitter_feature_selection(setup_data):
    df_iris_train = setup_data["df_iris_train"]
    df_iris_test = setup_data["df_iris_test"]

    dh = DataHandler(df_iris_train, df_iris_test, verbose=False)
    emitter = dh.train_test_emitter(
        "target", ["sepal_length_(cm)", "petal_length_(cm)", "petal_width_(cm)"]
    )
    emitter.select_predictors(["sepal_length_(cm)", "petal_length_(cm)"])
    assert np.allclose(
        emitter.emit_train_Xy()[0].to_numpy(),
        df_iris_train[["sepal length (cm)", "petal length (cm)"]].to_numpy(),
    )
    assert np.allclose(
        emitter.emit_test_Xy()[0].to_numpy(),
        df_iris_test[["sepal length (cm)", "petal length (cm)"]].to_numpy(),
    )
    assert np.allclose(
        emitter.emit_train_Xy()[1].to_numpy(), df_iris_train["target"].to_numpy()
    )
    assert np.allclose(
        emitter.emit_test_Xy()[1].to_numpy(), df_iris_test["target"].to_numpy()
    )

    df_house_train: pd.DataFrame = setup_data["df_house_train"]
    df_house_test: pd.DataFrame = setup_data["df_house_test"]

    dh = DataHandler(df_house_train, df_house_test, verbose=False)
    all_vars = ["SalePrice", "LotFrontage", "LotArea", "OverallQual", "LotShape"]
    emitter = dh.train_test_emitter("SalePrice", all_vars[1:])
    emitter.select_predictors(["OverallQual", "LotArea"])
    assert np.allclose(
        emitter.emit_train_Xy()[0].to_numpy(),
        df_house_train[all_vars].dropna()[["OverallQual", "LotArea"]].to_numpy(),
    )
    assert np.allclose(
        emitter.emit_test_Xy()[0].to_numpy(),
        df_house_test[all_vars].dropna()[["OverallQual", "LotArea"]].to_numpy(),
    )
    assert np.allclose(
        emitter.emit_train_Xy()[1].to_numpy(),
        df_house_train[all_vars].dropna()["SalePrice"].to_numpy(),
    )
    assert np.allclose(
        emitter.emit_test_Xy()[1].to_numpy(),
        df_house_test[all_vars].dropna()["SalePrice"].to_numpy(),
    )


def test_emitter_feature_selection_transform(setup_data):
    df_house_train = setup_data["df_house_train"]
    df_house_test = setup_data["df_house_test"]

    dh = DataHandler(df_house_train, df_house_test, verbose=False)


def test_datahandler_dropna_drop_highly_missing(setup_data):
    """Test DataEmitter with basic pipeline generation capabilities"""
    train_data = setup_data["df_house_train"]
    test_data = setup_data["df_house_test"]
    dh = DataHandler(train_data, test_data, verbose=True)

    dh.drop_highly_missing_vars(threshold=0.5)
    dh.dropna()
    assert "PoolQC" not in dh._working_df_train.columns

    de = dh.train_test_emitter("SalePrice", ["LotFrontage", "LotArea", "OverallQual"])

    assert "PoolQC" not in de._working_df_train.columns

    train_X, _, _, _ = de.emit_train_test_Xy()

    assert len(de._working_df_train) == len(train_X)


def test_datahandler_engineer_feature(setup_data):
    """Test DataEmitter with basic pipeline generation capabilities"""
    train_data = setup_data["df_house_train"]
    test_data = setup_data["df_house_test"]
    dh = DataHandler(train_data, test_data, verbose=False)

    dh.engineer_numeric_feature(
        "TotalSF",
        "TotalBsmtSF + 1stFlrSF + 2ndFlrSF",
    )

    train_df = dh.df_train()
    assert "TotalSF" in train_df.columns
    assert train_df["TotalSF"].equals(
        train_df["TotalBsmtSF"] + train_df["1stFlrSF"] + train_df["2ndFlrSF"]
    )

    test_df = dh.df_test()
    assert "TotalSF" in test_df.columns
    assert test_df["TotalSF"].equals(
        test_df["TotalBsmtSF"] + test_df["1stFlrSF"] + test_df["2ndFlrSF"]
    )

    de = dh.train_test_emitter("SalePrice", ["TotalSF", "LotArea", "OverallQual"])

    assert "TotalSF" in de._working_df_train.columns
    assert "TotalSF" in de._working_df_test.columns

    train_X, _, test_X, _ = de.emit_train_test_Xy()

    assert train_X["TotalSF"].equals(train_df["TotalSF"])
    assert test_X["TotalSF"].equals(test_df["TotalSF"])

    assert len(de._working_df_train) == len(train_X)

    dh.load_data_checkpoint()
    assert "TotalSF" not in dh.df_train().columns

    de = dh.train_test_emitter("SalePrice", ["LotArea", "OverallQual"])
    assert "TotalSF" not in de._working_df_train.columns

    train_X, _, test_X, _ = de.emit_train_test_Xy()

    assert len(dh._working_df_train) == len(train_X)
    assert len(dh._working_df_test) == len(test_X)


def test_datahandler_engineer_feature_transform(setup_data):
    train_data = setup_data["df_house_train"]
    test_data = setup_data["df_house_test"]
    dh = DataHandler(train_data, test_data, verbose=False)

    dh.engineer_numeric_feature(
        "TotalSF",
        "sqrt(TotalBsmtSF * log(1stFlrSF + 2ndFlrSF))",
    )

    extra_col = np.sqrt(
        train_data["TotalBsmtSF"]
        * np.log(train_data["1stFlrSF"] + train_data["2ndFlrSF"])
    )

    assert dh.df_train()["TotalSF"].equals(extra_col)

    de = dh.train_test_emitter("SalePrice", ["TotalSF", "LotArea", "OverallQual"])

    assert "TotalSF" in de._working_df_train.columns

    train_X, _, test_X, _ = de.emit_train_test_Xy()

    assert train_X["TotalSF"].equals(extra_col)

    assert len(de._working_df_train) == len(train_X)


def test_datahandler_engineer_categorical_feature(setup_data):
    train_data = setup_data["df_house_train"]
    test_data = setup_data["df_house_test"]
    dh = DataHandler(train_data, test_data, verbose=False)

    dh.engineer_categorical_feature(
        feature_name="1stFlrSF_quartiles",
        numeric_var="1stFlrSF",
        level_names=["Q1", "Q2", "Q3", "Q4"],
        thresholds=[882, 1087, 1391],
        leq=True,
    )

    train_df = dh.df_train()
    assert "1stFlrSF_quartiles" in train_df.columns
    test_df = dh.df_test()
    assert "1stFlrSF_quartiles" in test_df.columns

    de = dh.train_test_emitter("SalePrice", ["LotArea", "1stFlrSF_quartiles"])

    assert "1stFlrSF_quartiles" in de._working_df_train.columns
    assert "1stFlrSF_quartiles" in de._working_df_test.columns

    train_X, _, test_X, _ = de.emit_train_test_Xy()

    assert len(de._working_df_train) == len(train_X)

    dh.load_data_checkpoint()
    assert "1stFlrSF_quartiles" not in dh.df_train().columns

    de = dh.train_test_emitter("SalePrice", ["LotArea", "OverallQual"])
    assert "1stFlrSF_quartiles::Q2" not in de._working_df_train.columns

    train_X, _, test_X, _ = de.emit_train_test_Xy()

    assert len(dh._working_df_train) == len(train_X)
    assert len(dh._working_df_test) == len(test_X)
