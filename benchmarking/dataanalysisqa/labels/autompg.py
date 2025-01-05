from pathlib import Path
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

datasets_dir = Path(__file__).resolve().parent.parent / "datasets"

# import dataset
df = pd.read_csv(datasets_dir / "autompg.csv")
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train_idx = df_train.index
df_test_idx = df_test.index
del df_train, df_test


# Question 1 - What's the average miles per gallon ("mpg")?
def q1():
    keyword = "mean"
    answer = df["mpg"].mean()
    return f"{keyword}={answer:.3f}"


# Question 2 - Find the average miles per gallon for cars of model year 70.
def q2():
    keyword = "mean"
    answer = df[df["modelyear"] == 70]["mpg"].mean()
    return f"{keyword}={answer:.3f}"


# Question 3 - How many cars are of model year 75 or later? What's the mean horsepower of these cars?
def q3():
    keyword1 = "n_rows"
    keyword2 = "mean"
    n_cars = df[df["modelyear"] >= 75].shape[0]
    mean_hp = df[df["modelyear"] >= 75]["horsepower"].mean()
    return f"{keyword1}={n_cars}, {keyword2}={mean_hp:.3f}"


# Question 4 - Find the correlation between acceleration and weight. Report both the correlation coefficient and the p-value.
def q4():
    keyword1 = "corr"
    keyword2 = "pval"
    corr, pval = stats.pearsonr(df["acceleration"], df["weight"])
    return f"{keyword1}={corr:.3f}, {keyword2}={pval:.3f}"


# Question 5 - Make a linear regression model predicting the acceleration from weight. What is the coefficient for weight? What is the model's train R-squared?
def q5():
    keyword1 = "coef"
    keyword2 = "r2"

    df_train = df.loc[df_train_idx]
    df_test = df.loc[df_test_idx]

    X_train = df_train[["weight"]]
    y_train = df_train["acceleration"]

    X_test = df_test[["weight"]]
    y_test = df_test["acceleration"]

    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    model = sm.OLS(y_train, X_train).fit()
    coef = model.params["weight"]
    r2 = model.rsquared
    return f"{keyword1}={coef:.3f}, {keyword2}={r2:.3f}"


# Question 6 - Create a new variable named "heavy" with categories "heavy" and "light". An observation is "heavy" if its weight is at least 3200 and "light" otherwise. How many heavy observations are there?
def q6():
    global df
    keyword = "n_examples"
    df["heavy"] = np.where(df["weight"] >= 3200, "heavy", "light")
    n_heavy = df[df["heavy"] == "heavy"].shape[0]
    return f"{keyword}={n_heavy}"


# Question 7 - Is there a statistically significant difference in average miles per gallon between heavy and light vehicles?
def q7():
    keyword = "yes_or_no"
    pval = stats.ttest_ind(
        df[df["heavy"] == "heavy"]["mpg"], df[df["heavy"] == "light"]["mpg"]
    ).pvalue
    return f"{keyword}={'yes' if pval < 0.05 else 'no'}"


# Question 8 - Make a new variable, "powerful", with category "powerful" for those with "cylinder" of 8, and category "weak" for those with "cylinder" less than 8. How many "weak" vehicles are there?
def q8():
    global df
    keyword = "n_examples"
    df["powerful"] = np.where(df["cylinders"] == 8, "powerful", "weak")
    n_weak = df[df["powerful"] == "weak"].shape[0]
    return f"{keyword}={n_weak}"


# Question 9 - Are the variables "powerful" and "heavy" statistically independent?
def q9():
    keyword = "yes_or_no"
    pval = stats.chi2_contingency(pd.crosstab(df["powerful"], df["heavy"]))[1]
    return f"{keyword}={'yes' if pval > 0.05 else 'no'}"


# Question 10 - Is model year normally distributed?
def q10():
    keyword = "yes_or_no"
    pval = stats.shapiro(df["modelyear"])[1]
    return f"{keyword}={'yes' if pval > 0.05 else 'no'}"


def run_autompg():
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
    print(run_autompg())
