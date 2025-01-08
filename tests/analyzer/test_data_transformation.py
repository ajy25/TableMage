import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pathlib
import sys

parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent
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


def test_numeric_feature_engineering(setup_data):
    df: pd.DataFrame = setup_data["df_house"].copy()

    # test basic names
    analyzer = tm.Analyzer(df=df)

    analyzer.engineer_numeric_var(name="log_SalePrice", formula="log(SalePrice)")

    assert np.allclose(
        analyzer.df_all()["log_SalePrice"].to_numpy(),
        np.log(df["SalePrice"].to_numpy()),
    )
