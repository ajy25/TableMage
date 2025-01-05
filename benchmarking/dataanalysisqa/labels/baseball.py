from pathlib import Path
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import statsmodels.api as sm

datasets_dir = Path(__file__).resolve().parent.parent / "datasets"

# import dataset
df = pd.read_csv(datasets_dir / "baseball.csv")
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)


# Question 1 - What is the mean batting average? What is the standard deviation?
def q1():
    keyword1 = "mean"
    keyword2 = "std"
    answer1 = df["batting_average"].mean()
    answer2 = df["batting_average"].std()
    return f"{keyword1}={answer1:.3f}, {keyword2}={answer2:.3f}"


# Question 2 - Report whether or not batting average adheres to a normal distribution.
def q2():
    keyword = "yes_or_no"
    _, pval = stats.normaltest(df["batting_average"])
    return f"{keyword}={'yes' if pval > 0.05 else 'no'}"


# Question 3 - Is batting average correlated with salary?
def q3():
    keyword = "yes_or_no"
    corr, pval = stats.pearsonr(
        df["batting_average"], df["salary_in_thousands_of_dollars"]
    )
    return f"{keyword}={'yes' if pval < 0.05 else 'no'}"


# Question 4 - Min-max scale the salary. Report the new mean and standard deviation of the salary.
def q4():
    global df, df_train, df_test
    keyword1 = "mean"
    keyword2 = "std"
    scaler = MinMaxScaler()
    df_train["salary_in_thousands_of_dollars"] = scaler.fit_transform(
        df_train[["salary_in_thousands_of_dollars"]]
    )
    df_test["salary_in_thousands_of_dollars"] = scaler.transform(
        df_test[["salary_in_thousands_of_dollars"]]
    )
    df = pd.concat([df_train, df_test])
    answer1 = df["salary_in_thousands_of_dollars"].mean()
    answer2 = df["salary_in_thousands_of_dollars"].std()
    return f"{keyword1}={answer1:.3f}, {keyword2}={answer2:.3f}"


# Question 5 - Use linear regression to regress salary on batting_average. What is the test RMSE of the model?
def q5():
    keyword = "rmse"

    df_train_temp = df_train.dropna(
        subset=["salary_in_thousands_of_dollars", "batting_average"]
    )
    df_test_temp = df_test.dropna(
        subset=["salary_in_thousands_of_dollars", "batting_average"]
    )

    X_train = df_train_temp[["batting_average"]]
    y_train = df_train_temp["salary_in_thousands_of_dollars"]

    X_test = df_test_temp[["batting_average"]]
    y_test = df_test_temp["salary_in_thousands_of_dollars"]

    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    model = sm.OLS(y_train, X_train).fit()
    y_pred = model.predict(X_test)
    rmse = np.sqrt(((y_test - y_pred) ** 2).mean())
    return f"{keyword}={rmse:.3f}"


# Question 6 - Compute the interquartile range of batting_average. Identify outliers, based on 1.5 times the interquartile range. How many outliers are there?
def q6():
    keyword = "n_outliers"
    q1 = df["batting_average"].quantile(0.25)
    q3 = df["batting_average"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    n_outliers = df[
        (df["batting_average"] < lower_bound) | (df["batting_average"] > upper_bound)
    ].shape[0]
    return f"{keyword}={n_outliers}"


# Question 7 - Make a new variable called "hits_and_runs" that is the sum of number of runs and number of hits. What is the mean and kurtosis of this new variable?
def q7():
    global df, df_train, df_test
    keyword1 = "mean"
    keyword2 = "kurtosis"
    df["hits_and_runs"] = df["number_of_runs"] + df["number_of_hits"]
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    answer1 = df["hits_and_runs"].mean()
    answer2 = df["hits_and_runs"].kurtosis()
    return f"{keyword1}={answer1:.3f}, {keyword2}={answer2:.3f}"


# Question 8 - Standard scale "hits_and_runs". Find the median.
def q8():
    keyword = "median"
    scaler = StandardScaler()
    df_train["hits_and_runs"] = scaler.fit_transform(df_train[["hits_and_runs"]])
    df_test["hits_and_runs"] = scaler.transform(df_test[["hits_and_runs"]])
    df = pd.concat([df_train, df_test])
    answer = df["hits_and_runs"].median()
    return f"{keyword}={answer:.3f}"


# Question 9 - Among batting_average, on_base_percentage, number_of_runs, and number_of_hits, which variable is most highly correlated with salary_in_thousands_of_dollars?
def q9():
    keyword = "most_correlated"
    corr = df[
        [
            "batting_average",
            "on_base_percentage",
            "number_of_runs",
            "number_of_hits",
            "salary_in_thousands_of_dollars",
        ]
    ].corr()
    most_correlated = (
        corr["salary_in_thousands_of_dollars"]
        .abs()
        .sort_values(ascending=False)
        .index[1]
    )
    return f"{keyword}={most_correlated}"


# Question 10 - Undo all prior data transformations. What's the average salary?
def q10():
    global df, df_train, df_test
    keyword = "mean"
    df = pd.read_csv(datasets_dir / "baseball.csv")
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    answer = df["salary_in_thousands_of_dollars"].mean()
    return f"{keyword}={answer:.3f}"


def run_baseball():
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
    print(run_baseball())
