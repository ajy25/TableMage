from pathlib import Path
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from sklearn.metrics import root_mean_squared_error, r2_score

datasets_dir = Path(__file__).resolve().parent.parent / "datasets"

# import dataset
df = pd.read_csv(datasets_dir / "simulated_healthcare.csv")
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train_idx = df_train.index
df_test_idx = df_test.index
del df_train, df_test


# Question 1 - How many different blood types are there?
def q1():
    keyword = "n_types"
    answer = df["Blood Type"].nunique()

    answer = round(answer, 3)

    return f"{keyword}={answer:.3f}"


# Question 2 - How many different insurance providers are there?
def q2():
    keyword = "n_insurance_providers"
    answer = df["Insurance Provider"].nunique()

    answer = round(answer, 3)

    return f"{keyword}={answer:.3f}"


# Question 3 - Which insurance provider is associated with the highest average billing amount?
def q3():
    keyword = "insurance_provider"
    answer = df.groupby("Insurance Provider")["Billing Amount"].mean().idxmax()
    return f"{keyword}={answer}"


# Question 4 - Is there a statistically significant difference in average billing amount between males and females?
def q4():
    keyword = "yes_or_no"
    pval = stats.ttest_ind(
        df[df["Gender"] == "Male"]["Billing Amount"],
        df[df["Gender"] == "Female"]["Billing Amount"],
    ).pvalue
    return f"{keyword}={'yes' if pval < 0.05 else 'no'}"


# Question 5 - Use linear regression to predict billing amount from gender and insurance provider. What is the train R-squared of the model? What about the test RMSE?
def q5():
    keyword1 = "r2"
    keyword2 = "rmse"

    df_train = df.loc[df_train_idx]
    df_test = df.loc[df_test_idx]

    X_train = df_train[["Gender", "Insurance Provider"]]
    y_train = df_train["Billing Amount"]

    X_test = df_test[["Gender", "Insurance Provider"]]
    y_test = df_test["Billing Amount"]

    X_train = pd.get_dummies(X_train, drop_first=True).astype(float)
    X_train = sm.add_constant(X_train)

    X_test = pd.get_dummies(X_test, drop_first=True).astype(float)
    X_test = sm.add_constant(X_test)

    model = sm.OLS(y_train, X_train).fit()
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    r2 = r2_score(y_train, y_pred_train)
    rmse = root_mean_squared_error(y_test, y_pred_test)

    r2 = round(r2, 3)
    rmse = round(rmse, 3)

    return f"{keyword1}={r2:.3f}, {keyword2}={rmse:.3f}"


# Question 6 - Are the variables blood type and gender statistically independent?
def q6():
    keyword = "yes_or_no"
    pval = stats.chi2_contingency(pd.crosstab(df["Blood Type"], df["Gender"]))[1]
    return f"{keyword}={'yes' if pval > 0.05 else 'no'}"


# Question 7 - Regress billing amount on age with linear regression. What is the coefficient associated with age? What is the intercept value?
def q7():
    keyword1 = "coef"
    keyword2 = "intercept"

    df_train = df.loc[df_train_idx]

    X = df_train[["Age"]]
    y = df_train["Billing Amount"]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    age_coef = model.params["Age"]
    intercept = model.params["const"]

    age_coef = round(age_coef, 3)
    intercept = round(intercept, 3)

    return f"{keyword1}={age_coef:.3f}, {keyword2}={intercept:.3f}"


# Question 8 - Min-max scale the billing amount. What is the variance of the billing amount?
def q8():
    global df
    keyword = "variance"
    scaler = MinMaxScaler()

    df_train = df.loc[df_train_idx]
    df_test = df.loc[df_test_idx]

    df_train["Billing Amount"] = scaler.fit_transform(
        df_train["Billing Amount"].values.reshape(-1, 1)
    )
    df_test["Billing Amount"] = scaler.transform(
        df_test["Billing Amount"].values.reshape(-1, 1)
    )
    df = pd.concat([df_train, df_test]).loc[df.index]

    var = df["Billing Amount"].var()
    var = round(var, 3)

    return f"{keyword}={var:.3f}"


# Question 9 - What is the average billing amount?
def q9():
    keyword = "mean"
    answer = df["Billing Amount"].mean()

    answer = round(answer, 3)

    return f"{keyword}={answer:.3f}"


# Question 10 - Which medical condition is associated with the highest billing amount? What is the average?
def q10():
    keyword1 = "medical_condition"
    keyword2 = "mean"
    answer1 = df.groupby("Medical Condition")["Billing Amount"].mean().idxmax()

    answer2 = df.groupby("Medical Condition")["Billing Amount"].mean().max()
    answer2 = round(answer2, 3)
    return f"{keyword1}={answer1}, {keyword2}={answer2:.3f}"


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
    }


if __name__ == "__main__":
    print(get_labels())
