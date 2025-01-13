import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pathlib
import sys

parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(parent_dir))

from tablemage._src.data.datahandler import (
    DataHandler,
)


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


def test_dataemitter_sklearn_transformer_basic(setup_data):
    """Test DataEmitter with basic sklearn transformer generation capabilities"""
    train_data = setup_data["df_house_train"]
    test_data = setup_data["df_house_test"]
    dh = DataHandler(train_data, test_data, verbose=False)

    dh.scale(
        include_vars=["GrLivArea", "YearBuilt", "OverallQual", "LotArea", "SalePrice"],
        strategy="standardize",
    )
    dh.onehot(["MSZoning"])

    de = dh.train_test_emitter(
        y_var="SalePrice",
        X_vars=["GrLivArea", "YearBuilt", "OverallQual", "LotArea", "LotShape"],
    )

    de.emit_train_test_Xy()

    emitted_test_Xy = de.emit_test_Xy()
    emitted_df = emitted_test_Xy[0]

    transformer = de.sklearn_preprocessing_transformer()
    transformer_df = transformer.transform(test_data)

    assert "SalePrice" not in transformer_df.columns
    assert np.allclose(emitted_df.values, transformer_df.values, atol=1e-5)
