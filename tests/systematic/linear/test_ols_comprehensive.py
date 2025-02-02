import pytest
import pandas as pd
import numpy as np
import pathlib
import sys
from sklearn.model_selection import train_test_split


parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(parent_dir))


import tablemage as tm


@pytest.fixture
def setup_data() -> dict:
    df_house = pd.read_csv(
        parent_dir / "demo" / "regression" / "house_price_data" / "data.csv"
    )
    df_house["SalePrice"] = df_house["SalePrice"].astype(float) / 1000

    df_house["IsExpensive"] = (df_house["SalePrice"] > 150).astype(int)

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


def test_ols_comprehensive_basic(setup_data):
    """Tests basic functionality of the ols method in a comprehensive manner"""
    df_house = setup_data["df_house"]
    analyzer = tm.Analyzer(df_house, test_size=0.2, split_seed=42)

    # test basic
    report = analyzer.ols(
        "SalePrice",
        predictors=[
            "OverallQual",
            "GrLivArea",
            "GarageCars",
            "TotalBsmtSF",
            "FullBath",
            "YearBuilt",
            "YearRemodAdd",
            "MSZoning",
            "OverallCond",
            "LotArea",
            "1stFlrSF",
            "GarageArea",
            "2ndFlrSF",
        ],
    )

    # ensure coefficients can be produced
    report.coefs(format="coef(se)|pval")
    report.coefs(format="coef|se|pval")
    report.coefs(format="coef(ci)|pval")
    report.coefs(format="coef|ci_low|ci_high|pval")

    # reduce the model
    backward_reduced_report = report.step("backward", criteria="aic")
    forward_reduced_report = report.step("forward", criteria="aic")
    bidirectional_reduced_report = report.step("both", criteria="aic")

    # ensure number of predictors is not greater than the original model
    assert len(backward_reduced_report._predictors) <= len(report._predictors)
    assert len(forward_reduced_report._predictors) <= len(report._predictors)
    assert len(bidirectional_reduced_report._predictors) <= len(report._predictors)

    # ensure coefficients can be produced
    backward_reduced_report.coefs(format="coef(se)|pval")
    backward_reduced_report.coefs(format="coef|se|pval")
    backward_reduced_report.coefs(format="coef(ci)|pval")
    backward_reduced_report.coefs(format="coef|ci_low|ci_high|pval")

    forward_reduced_report.coefs(format="coef(se)|pval")
    forward_reduced_report.coefs(format="coef|se|pval")
    forward_reduced_report.coefs(format="coef(ci)|pval")
    forward_reduced_report.coefs(format="coef|ci_low|ci_high|pval")

    bidirectional_reduced_report.coefs(format="coef(se)|pval")
    bidirectional_reduced_report.coefs(format="coef|se|pval")
    bidirectional_reduced_report.coefs(format="coef(ci)|pval")
    bidirectional_reduced_report.coefs(format="coef|ci_low|ci_high|pval")

    # ensure both lr test and f test work
    backward_reduced_report.test_lr(alternative_report=report)
    backward_reduced_report.test_partialf(alternative_report=report)

    forward_reduced_report.test_lr(alternative_report=report)
    forward_reduced_report.test_partialf(alternative_report=report)

    bidirectional_reduced_report.test_lr(alternative_report=report)
    bidirectional_reduced_report.test_partialf(alternative_report=report)

    # plot diagnostics
    report.plot_diagnostics("train")
    report.plot_diagnostics("test")

    backward_reduced_report.plot_diagnostics("train")
    backward_reduced_report.plot_diagnostics("test")

    forward_reduced_report.plot_diagnostics("train")
    forward_reduced_report.plot_diagnostics("test")

    bidirectional_reduced_report.plot_diagnostics("train")
    bidirectional_reduced_report.plot_diagnostics("test")

    report.plot_residuals_hist("train")
    report.plot_residuals_hist("test")

    report.plot_residuals_vs_var(predictor="OverallQual", dataset="train")
    report.plot_residuals_vs_var(predictor="OverallQual", dataset="test")
