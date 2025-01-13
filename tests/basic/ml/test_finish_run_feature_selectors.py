import pathlib
import sys
import pandas as pd
import pytest
import numpy as np


parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(parent_dir))


import tablemage as tm


SAMPLE_SIZE = 200


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
    analyzer = tm.Analyzer(setup_data["df_house_mini"], test_size=0.4, verbose=False)

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
            "YearBuilt",
        ],
        feature_selectors=[
            tm.fs.KBestFSR(
                scorer="f_regression",
                k=3,
            ),
            tm.fs.BorutaFSR(),
            tm.fs.LassoFSR(max_n_features=3),
        ],
    )

    report._to_dict()

    assert isinstance(report.fs_report().top_features(), list)
