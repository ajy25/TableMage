from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error


datasets_dir = Path(__file__).resolve().parent.parent / "datasets"

# import dataset
df = pd.read_csv(datasets_dir / "abalone.csv")
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train_idx = df_train.index
df_test_idx = df_test.index
del df_train, df_test


# Question 1 - How many different classes of "Sex" are there?
def q1():
    keyword = "n_sex_classes"
    answer = len(df["Sex"].unique())
    answer = round(answer, 3)
    return f"{keyword}={answer:.3f}"


# Question 2 - Find the mean diameter.
def q2():
    keyword = "mean"
    answer = df["Diameter"].mean()
    answer = round(answer, 3)
    return f"{keyword}={answer:.3f}"


# Question 3 - Compute the variance of shucked weight.
def q3():
    keyword = "variance"
    answer = df["Shucked weight"].var()
    answer = round(answer, 3)
    return f"{keyword}={answer:.3f}"


# Question 4 - What is the average diameter for those with "Sex" set to "M"?
def q4():
    keyword = "mean"
    answer = df[df["Sex"] == "M"]["Diameter"].mean()
    answer = round(answer, 3)
    return f"{keyword}={answer:.3f}"


# Question 5 - Find the correlation between diameter and rings. Report the correlation and the p-value.
def q5():
    keyword1 = "corr"
    keyword2 = "pval"
    correlation, p_value = stats.pearsonr(df["Diameter"], df["Rings"])
    correlation = round(correlation, 3)
    p_value = round(p_value, 3)
    return f"{keyword1}={correlation:.3f}, {keyword2}={p_value:.3f}"


# Question 6 - Is the diameter normally distributed?
def q6():
    keyword = "yes_or_no"
    _, p_value_normality = stats.shapiro(df["Diameter"])

    # check other methods so that this assessment is flexible to method choice
    _, p_value_k2 = stats.kstest(df["Diameter"], "norm")
    _, p_value_normaltest = stats.normaltest(df["Diameter"])

    assert (
        (p_value_normality > 0.05) == (p_value_k2 > 0.05) == (p_value_normaltest > 0.05)
    )

    answer = "no" if (p_value_normality <= 0.05) else "yes"
    return f"{keyword}={answer}"


# Question 7 - Is there a statistically significant difference in average "Diameter" between the "Sex" categories?
def q7():
    keyword = "yes_or_no"
    _, anova_p_value = stats.f_oneway(
        df[df["Sex"] == "M"]["Diameter"],
        df[df["Sex"] == "F"]["Diameter"],
        df[df["Sex"] == "I"]["Diameter"],
    )

    # check other methods so that this assessment is flexible to method choice
    _, p_value_k2 = stats.kruskal(
        df[df["Sex"] == "M"]["Diameter"],
        df[df["Sex"] == "F"]["Diameter"],
        df[df["Sex"] == "I"]["Diameter"],
    )

    assert (anova_p_value <= 0.05) == (p_value_k2 <= 0.05)

    answer = "yes" if anova_p_value <= 0.05 else "no"
    return f"{keyword}={answer}"


# Question 8 - Create a new variable, "Area", which is the product of "Length" and "Height". Report its median.
def q8():
    global df
    keyword = "median"
    df["Area"] = df["Length"] * df["Height"]
    answer = df["Area"].median()
    answer = round(answer, 3)
    return f"{keyword}={answer:.3f}"


# Question 9 - Based on "Area", create a new variable named "LargeArea" with category "Yes" if "Area" is at least 0.0775, "No" otherwise. Find the number of examples with "Yes" for "LargeArea".
def q9():
    global df
    keyword = "n_yes"
    df["LargeArea"] = df["Area"].apply(lambda x: "Yes" if x >= 0.0775 else "No")
    answer = df[df["LargeArea"] == "Yes"].shape[0]
    answer = round(answer, 3)
    return f"{keyword}={answer:.3f}"


# Question 10 - Fit a linear regression model to predict shucked weight with "LargeArea" and "Area". Report the test mean absolute error.
def q10():
    keyword = "mae"

    df_train = df.loc[df_train_idx]
    df_test = df.loc[df_test_idx]

    # Fit a linear regression model using statsmodels
    model = smf.ols(
        formula='Q("Shucked weight") ~ Area + C(LargeArea)', data=df_train
    ).fit()

    # Make predictions on the test set
    test_predictions = model.predict(df_test)

    # Calculate the test mean absolute error
    test_mae = mean_absolute_error(df_test["Shucked weight"], test_predictions)
    test_mae = round(test_mae, 3)
    return f"{keyword}={test_mae:.3f}"


# Question 11 - Are "LargeArea" and "Sex" statistically independent?
def q11():
    keyword = "yes_or_no"

    # Create a contingency table for 'LargeArea' and 'Sex'
    contingency_table = pd.crosstab(df["LargeArea"], df["Sex"])

    # Perform the chi-squared test for independence
    _, p_value_independence, _, _ = stats.chi2_contingency(contingency_table)

    answer = "no" if p_value_independence <= 0.05 else "yes"
    return f"{keyword}={answer}"


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
    }


if __name__ == "__main__":
    print(get_labels())
