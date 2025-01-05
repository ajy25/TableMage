from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.metrics import accuracy_score, f1_score
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import r2_score

datasets_dir = Path(__file__).resolve().parent.parent / "datasets"

# import dataset
df = pd.read_csv(datasets_dir / "wine.csv")
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train_idx = df_train.index
df_test_idx = df_test.index
del df_train, df_test


# Question 1 - Compute the mean and standard deviation for "alcohol".
def q1():
    keyword1 = "mean"
    keyword2 = "std"

    mean = df["alcohol"].mean()
    std = df["alcohol"].std()
    return f"{keyword1}={mean:.3f}, {keyword2}={std:.3f}"


# Question 2 - Compute the mean and standard deviation for "malic_acid".
def q2():
    keyword1 = "mean"
    keyword2 = "std"

    mean_malic_acid = df["malic_acid"].mean()
    std_malic_acid = df["malic_acid"].std()
    return f"{keyword1}={mean_malic_acid:.3f}, {keyword2}={std_malic_acid:.3f}"


# Question 3 - What is the skew and kurthosis of "alcohol"?
def q3():
    keyword1 = "skew"
    keyword2 = "kurtosis"

    skew_alcohol = skew(df["alcohol"])
    kurtosis_alcohol = kurtosis(
        df["alcohol"], fisher=True
    )  # Fisher=True for excess kurtosis
    return f"{keyword1}={skew_alcohol:.3f}, {keyword2}={kurtosis_alcohol:.3f}"


# Question 4 - Compute the correlation between "alcohol" and "malic_acid".
def q4():
    keyword = "corr"
    correlation = df["alcohol"].corr(df["malic_acid"])
    return f"{keyword}={correlation:.3f}"


# Question 5 - Fit a logistic regression model to predict "wine_class" from "alcohol", "malic_acid", and "flavanoids". Report the test F1 score and accuracy.
def q5():
    keyword1 = "f1"
    keyword2 = "acc"

    df_train = df.loc[df_train_idx]
    df_test = df.loc[df_test_idx]

    X_train = df_train[["alcohol", "malic_acid", "flavanoids"]]
    y_train = df_train["wine_class"]

    X_test = df_test[["alcohol", "malic_acid", "flavanoids"]]
    y_test = df_test["wine_class"]

    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    model = sm.MNLogit(y_train, X_train).fit()

    # Make predictions on the test set
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(
        y_pred_probs.values, axis=1
    )  # Convert probabilities to class predictions

    # Calculate accuracy and F1 score
    f1 = f1_score(y_test, y_pred, average="weighted")
    accuracy = accuracy_score(y_test, y_pred)

    return f"{keyword1}={f1:.3f}, {keyword2}={accuracy:.3f}"


# Question 6 - Engineer a new variable, "meaningless", that is defined as ("proline" - "alcohol" * "malic_acid"). Find its median.
def q6():
    global df
    keyword = "median"

    df["meaningless"] = df["proline"] - (df["alcohol"] * df["malic_acid"])

    # Calculate the median of "meaningless"
    median_meaningless = df["meaningless"].median()
    return f"{keyword}={median_meaningless:.3f}"


# Question 7 - What is the third largest value of "alcohol"?
def q7():
    keyword = "value"
    third_largest_alcohol = df["alcohol"].nlargest(3).iloc[-1]
    return f"{keyword}={third_largest_alcohol:.3f}"


# Question 8 - How many of each "wine_class" class are there in the dataset?
def q8():
    keyword1 = "class_0_count"
    keyword2 = "class_1_count"
    keyword3 = "class_2_count"

    count_0 = df["wine_class"].value_counts().get(0, 0)
    count_1 = df["wine_class"].value_counts().get(1, 0)
    count_2 = df["wine_class"].value_counts().get(2, 0)
    return (
        f"{keyword1}={count_0:.3f}, {keyword2}={count_1:.3f}, {keyword3}={count_2:.3f}"
    )


# Question 9 - Regress "meaningless" on "flavanoids" with linear regression. Report the test R-squared.
def q9():
    keyword = "r2"

    df_train = df.loc[df_train_idx]
    df_test = df.loc[df_test_idx]

    X_train = df_train[["flavanoids"]]
    y_train = df_train["meaningless"]

    X_test = df_test[["flavanoids"]]
    y_test = df_test["meaningless"]

    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    # Fit the linear regression model using statsmodels
    linear_model = sm.OLS(y_train, X_train).fit()

    # Make predictions on the test set
    y_pred = linear_model.predict(X_test)

    # Calculate the R-squared value on the test set
    r_squared = r2_score(y_test, y_pred)
    return f"{keyword}={r_squared:.3f}"


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
    }


if __name__ == "__main__":
    print(get_labels())
