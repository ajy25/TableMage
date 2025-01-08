from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import scipy.stats as stats

datasets_dir = Path(__file__).resolve().parent.parent / "datasets"

# import dataset
df = pd.read_csv(datasets_dir / "house_prices.csv")
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train_idx = df_train.index
df_test_idx = df_test.index
del df_train, df_test


# Question 1 - Compute the average "SalePrice" along with the standard deviation.
def q1():
    keyword1 = "mean"
    keyword2 = "std"

    mean = df["SalePrice"].mean()
    std = df["SalePrice"].std()

    mean = round(mean, 3)
    std = round(std, 3)

    return f"{keyword1}={mean:.3f}, {keyword2}={std:.3f}"


# Question 2 - Create a new variable, "TotalSF", which is defined as the sum of "1stFlrSF" and "2ndFlrSF". Find this new variable's mean.
def q2():
    global df
    keyword = "mean"

    df["TotalSF"] = df["1stFlrSF"] + df["2ndFlrSF"]
    mean = df["TotalSF"].mean()

    mean = round(mean, 3)

    return f"{keyword}={mean:.3f}"


# Question 3 - Impute missing values of "GarageYrBlt" with the median. Report its new mean.
def q3():
    global df
    keyword = "mean"

    df_train = df.loc[df_train_idx]

    # Get the median of the train dataset
    garage_median_train = df_train["GarageYrBlt"].median()  # Skips NA values by default

    # Impute on the entire dataset
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(garage_median_train, inplace=False)

    mean = df["GarageYrBlt"].mean()

    mean = round(mean, 3)

    return f"{keyword}={mean:.3f}"


# Question 4 - Which variable has the highest missingness? Report its name and its number of missing values.
def q4():
    keyword1 = "name"
    keyword2 = "n_missing"

    missing_counts = df.isnull().sum()
    max_missing_variable = missing_counts.idxmax()
    max_missing_value = missing_counts.max()

    max_missing_value = round(max_missing_value, 3)

    return f"{keyword1}={max_missing_variable}, {keyword2}={max_missing_value:.3f}"


# Question 5 - Regress "SalePrice" on "TotalSF" with linear regression. What is the value for the coefficient of "TotalSF"? Is the coefficient statistically significant? What is the intercept value?
def q5():
    keyword1 = "totalsf_coef"
    keyword2 = "totalsf_significant_yes_or_no"
    keyword3 = "intercept"

    df_train = df.loc[df_train_idx]
    df_test = df.loc[df_test_idx]

    X_train = df_train[["TotalSF"]]
    y_train = df_train["SalePrice"]

    X_train = sm.add_constant(X_train)

    model_train = sm.OLS(y_train, X_train).fit()

    # Extract coefficients
    coefficient_total_sf_train = model_train.params["TotalSF"]
    coefficient_total_sf_train = round(coefficient_total_sf_train, 3)

    p_value_total_sf_train = model_train.pvalues["TotalSF"]
    intercept_train = model_train.params["const"]

    intercept_train = round(intercept_train, 3)

    answer2 = "yes" if p_value_total_sf_train <= 0.05 else "no"
    return f"{keyword1}={coefficient_total_sf_train:.3f}, {keyword2}={answer2}, {keyword3}={intercept_train:.3f}"


# Question 6 - Regress "SalePrice" on "TotalSF", "LotShape", and "GarageArea" with linear regression. Report the train R-squared and the test RMSE.
def q6():
    keyword1 = "r2"
    keyword2 = "rmse"

    df_train = df.loc[df_train_idx]
    df_test = df.loc[df_test_idx]

    model_train = smf.ols(
        formula="SalePrice ~ TotalSF + C(LotShape) + GarageArea", data=df_train
    ).fit()  # Constant automatically added

    train_r2 = model_train.rsquared
    train_r2 = round(train_r2, 3)

    test_predictions = model_train.predict(df_test)

    # Calculate RMSE
    y_test = df_test["SalePrice"]

    test_rmse = np.sqrt(np.mean((test_predictions - y_test) ** 2))
    test_rmse = round(test_rmse, 3)

    return f"{keyword1}={train_r2:.3f}, {keyword2}={test_rmse:.3f}"


# Question 7 - Is there a statistically significant difference in "SalePrice" between the values of "LotShape"?
def q7():
    keyword = "yes_or_no"

    # Drop Missing values
    filtered_data = df.dropna(subset=["SalePrice", "LotShape"])

    # Get SalePrice of Different LotShape values
    shape_reg = filtered_data[filtered_data["LotShape"] == "Reg"]["SalePrice"]
    shape_IR1 = filtered_data[filtered_data["LotShape"] == "IR1"]["SalePrice"]
    shape_IR2 = filtered_data[filtered_data["LotShape"] == "IR2"]["SalePrice"]
    shape_IR3 = filtered_data[filtered_data["LotShape"] == "IR3"]["SalePrice"]

    # Perform one-way ANOVA
    statistic, pval = stats.f_oneway(shape_reg, shape_IR1, shape_IR2, shape_IR3)

    # Ensure other methods return same y/n answer
    _, kruskal_p_value = stats.kruskal(shape_reg, shape_IR1, shape_IR2, shape_IR3)

    assert (pval > 0.05) == (kruskal_p_value > 0.05)

    answer = "yes" if pval <= 0.05 else "no"

    return f"{keyword}={answer}"


# Question 8 - Compute the correlation between "SalePrice" and "TotalSF". Report the correlation as well as the p-value.
def q8():
    keyword1 = "corr"
    keyword2 = "pval"

    correlation, p_value = stats.pearsonr(df["SalePrice"], df["TotalSF"])

    return f"{keyword1}={correlation:.3f}, {keyword2}={p_value:.3f}"


# Question 9 - Is the distribution of "SalePrice" normal?
def q9():
    keyword = "yes_or_no"
    SalePriceCol = df["SalePrice"]

    # Perform a Shapiro-Wilk test for normality. Null is that the data is normally distributed
    _, shapiro_p_value = stats.shapiro(SalePriceCol)

    # Ensure other methods return same y/n answer
    _, normaltest_p_value = stats.normaltest(SalePriceCol)
    _, kstest_p_value = stats.kstest(SalePriceCol, "norm")

    assert (
        (shapiro_p_value > 0.05)
        == (normaltest_p_value > 0.05)
        == (kstest_p_value > 0.05)
    )

    answer = "no" if (shapiro_p_value <= 0.05) else "yes"
    return f"{keyword}={answer}"


# Question 10 - Engineer a new variable, "PriceRange", with values "Low", "Medium", and "High", based on "SalePrice". "Low" is defined as having "SalePrice" below 100,000. "Medium" is defined as having "SalePrice" at least 100,000 but below 300,000. "High" is defined as having "SalePrice" at least 300,000. Find the average "SalePrice" among houses considered in the "Medium" price range.
def q10():
    global df
    keyword = "mean"

    df["PriceRange"] = df["SalePrice"].apply(
        lambda price: (
            "Low" if price < 100000 else ("Medium" if price < 300000 else "High")
        )
    )
    medium_price_avg_lambda = df[df["PriceRange"] == "Medium"]["SalePrice"].mean()

    medium_price_avg_lambda = round(medium_price_avg_lambda, 3)

    return f"{keyword}={medium_price_avg_lambda:.3f}"


# Question 11 - Report the value counts of "PriceRange".
def q11():
    keyword1 = "n_low"
    keyword2 = "n_medium"
    keyword3 = "n_high"

    low = df["PriceRange"].value_counts().get("Low", 0)
    medium = df["PriceRange"].value_counts().get("Medium", 0)
    high = df["PriceRange"].value_counts().get("High", 0)

    low = round(low, 3)
    medium = round(medium, 3)
    high = round(high, 3)

    return f"{keyword1}={low:.3f}, {keyword2}={medium:.3f}, {keyword3}={high:.3f}"


# Question 12 - Regress "SalePrice" on "TotalSF", "GarageYrBlt", and "GarageArea" with linear regression. Report the train R-squared and the test RMSE.
def q12():
    keyword1 = "r2"
    keyword2 = "rmse"

    df_train = df.loc[df_train_idx]
    df_test = df.loc[df_test_idx]

    model_train = smf.ols(
        formula="SalePrice ~ TotalSF + GarageYrBlt + GarageArea", data=df_train
    ).fit()

    train_r2 = model_train.rsquared

    test_predictions = model_train.predict(df_test)

    # Calculate RMSE
    y_test = df_test["SalePrice"]
    test_rmse = np.sqrt(np.mean((test_predictions - y_test) ** 2))

    train_r2 = round(train_r2, 3)
    test_rmse = round(test_rmse, 3)

    return f"{keyword1}={train_r2:.3f}, {keyword2}={test_rmse:.3f}"


def get_labels():
    return {
        1: q1(),
        2: q2(),
        3: q3(),
        4: q4(),
        5: q5(),
        6: q6(),
        7: q7(),
        8: q8(),
        9: q9(),
        10: q10(),
        11: q11(),
        12: q12(),
    }


if __name__ == "__main__":
    print(get_labels())
