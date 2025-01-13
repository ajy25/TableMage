import pathlib
import sys
import pandas as pd
import pytest
import numpy as np


parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(parent_dir))


import tablemage as tm

"""
Here we test the Woolridge's card dataset.
Package methods are tested against R-based "ground truth" estimates.
"""


@pytest.fixture
def setup_data():
    card_df = pd.read_csv(parent_dir / "demo" / "causal" / "data" / "card.csv")
    card_df["educ_binary"] = card_df["educ"].apply(lambda x: 0 if x < 13 else 1)
    return {
        "card": card_df,
    }


def test_naive(setup_data):
    data = setup_data["card"]
    analyzer = tm.Analyzer(data)
    causal_model = analyzer.causal(
        treatment="educ_binary", outcome="lwage", confounders=[]
    )
    report = causal_model.estimate_ate(method="naive")
    naive_estimate = report._estimate
    assert np.isclose(naive_estimate, 0.1942883, atol=1e-4)
    assert np.isclose(report._estimate_se, 0.01579299, atol=1e-4)


def test_ols(setup_data):
    data = setup_data["card"]
    analyzer = tm.Analyzer(data)
    causal_model = analyzer.causal(
        treatment="educ_binary",
        outcome="lwage",
        confounders=[
            "exper",
            "expersq",
            "black",
            "smsa",
            "south",
            "smsa66",
            "reg662",
            "reg663",
            "reg664",
            "reg665",
            "reg666",
            "reg667",
            "reg668",
            "reg669",
        ],
    )
    report = causal_model.estimate_ate(
        method="outcome_regression", robust_se="nonrobust"
    )
    ols_estimate = report._estimate
    assert np.isclose(ols_estimate, 0.2384783, atol=1e-4)
    assert np.isclose(report._estimate_se, 0.0172869, atol=1e-4)


def test_weighted_reg(setup_data):
    data = setup_data["card"]
    analyzer = tm.Analyzer(data)
    causal_model = analyzer.causal(
        treatment="educ_binary",
        outcome="lwage",
        confounders=[
            "exper",
            "expersq",
            "black",
            "smsa",
            "south",
            "smsa66",
            "reg662",
            "reg663",
            "reg664",
            "reg665",
            "reg666",
            "reg667",
            "reg668",
            "reg669",
        ],
    )
    report = causal_model.estimate_ate(
        method="ipw_weighted_regression", robust_se="HC0"
    )
    weighted_estimate = report._estimate
    assert np.isclose(weighted_estimate, 0.2286431, atol=1e-4)
    # unable to perfectly reproduce SE with statsmodels, but close enough
    assert np.isclose(report._estimate_se, 0.0214608, atol=1e-4)

    report = causal_model.estimate_att(
        method="ipw_weighted_regression", robust_se="HC0"
    )
    weighted_estimate = report._estimate
    # unable to perfectly reproduce results with statsmodels, but close enough
    assert np.isclose(weighted_estimate, 0.2316082, atol=1e-4)
    assert np.isclose(report._estimate_se, 0.0198692, atol=1e-4)
