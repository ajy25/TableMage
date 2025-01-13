import pathlib
import sys
import pandas as pd
import pytest
import numpy as np
import statsmodels.api as sm


parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(parent_dir))


import tablemage as tm

SAMPLE_SIZE = 100


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

    df_mtcars = pd.read_csv(
        parent_dir / "demo" / "regression" / "mtcars_data" / "mtcars.csv"
    )

    return {
        "df_house": df_house,
        "df_house_mini": df_house_mini,
        "df_mtcars": df_mtcars,
    }


def math_linear_regression(X, y):
    # Add a constant term (intercept) to the predictors
    X = np.column_stack((np.ones(X.shape[0]), X))

    # Compute the coefficients using the normal equation: (X.T * X)^-1 * X.T * y
    coefficients = np.linalg.inv(X.T @ X) @ X.T @ y
    return coefficients


# def statsmodels_regression(data)


def tm_linear_regression(data):
    analyzer = tm.Analyzer(data, test_size=0.0, verbose=False)
    report = analyzer.ols(target="mpg", predictors=["cyl", "disp"])
    extract_model = report._model.estimator
    return (
        extract_model.params.iloc[0],
        extract_model.params.iloc[1],
        extract_model.params.iloc[2],
    )


# Test case to compare the outputs of our OLS wrapper and statsmodels OLS
def test_coefficients(setup_data):
    df_mtcars = setup_data["df_mtcars"]

    # Get the output from the custom Python function
    tm_b0, tm_b1, tm_b2 = tm_linear_regression(df_mtcars)

    # Define the predictors and the response variable
    X = df_mtcars[["cyl", "disp"]]
    y = df_mtcars["mpg"]

    # Add a constant term for the intercept
    X = sm.add_constant(X)

    # Fit the model using statsmodels
    model = sm.OLS(y, X).fit()

    # Get the first coefficient (the intercept)
    sm_b0 = model.params.iloc[0]
    sm_b1 = model.params.iloc[1]
    sm_b2 = model.params.iloc[2]

    # Use assert to compare the outputs
    assert np.isclose(tm_b0, sm_b0)
    assert np.isclose(tm_b1, sm_b1)
    assert np.isclose(tm_b2, sm_b2)


# Test case for partialf in OLS
def test_partialf_ols(setup_data):
    df_mtcars = setup_data["df_mtcars"]

    # r code for performing a partial F-test:
    # data(mtcars)
    # red.mod <- lm(hp ~ am + qsec, data = mtcars)
    # full.mod <- lm(hp ~ am + qsec + gear, data = mtcars)
    # anova(red.mod, full.mod)
    # output: F = 0.4416, p-val = 0.5118

    r_f_stat, r_p_val = 0.4416, 0.5118

    analyzer = tm.Analyzer(df_mtcars, test_size=0.0, verbose=False)
    lm_report_red = analyzer.ols(target="hp", predictors=["am", "qsec"])
    lm_report_full = analyzer.ols(target="hp", predictors=["am", "qsec", "gear"])

    tm_pval1 = lm_report_full.test_partialf(lm_report_red).pval()
    tm_pval2 = lm_report_red.test_partialf(lm_report_full).pval()

    epsilon = 0.0001
    assert np.isclose(tm_pval1, tm_pval2)
    assert np.isclose(tm_pval1, r_p_val, atol=epsilon)
    # To avoid propagation of error epsilon:
    assert np.isclose(tm_pval2, r_p_val, atol=epsilon)

    tm_fstat1 = lm_report_full.test_partialf(lm_report_red).statistic()
    tm_fstat2 = lm_report_red.test_partialf(lm_report_full).statistic()

    assert np.isclose(tm_fstat1, tm_fstat2)
    assert np.isclose(tm_fstat1, r_f_stat, atol=epsilon)
    # To avoid propagation of error epsilon:
    assert np.isclose(tm_fstat2, r_f_stat, atol=epsilon)


# Run the test
if __name__ == "__main__":
    pytest.main([__file__])
