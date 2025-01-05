from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import shapiro
from scipy.stats import f_oneway
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error
from scipy.stats import chi2_contingency


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
    return f"{keyword}={answer:.3f}"


# Question 2 - Find the mean diameter.
def q2():
    keyword = "mean"
    answer = df["Diameter"].mean()
    return f"{keyword}={answer:.3f}"


# Question 3 - Compute the variance of shucked weight.
def q3():
    keyword = "variance"
    answer = df["Shucked weight"].var()
    return f"{keyword}={answer:.3f}"


# Question 4 - What is the average diameter for those with "Sex" set to "M"?
def q4():
    keyword = "mean"
    answer = df[df["Sex"] == "M"]["Diameter"].mean()
    return f"{keyword}={answer:.3f}"


# Question 5 - Find the correlation between diameter and rings. Report the correlation and the p-value.
def q5():
    keyword1 = "corr"
    keyword2 = "pval"
    correlation, p_value = pearsonr(df["Diameter"], df["Rings"])
    return f"{keyword1}={correlation:.3f}, {keyword2} = {p_value:.3f}"


# Question 6 - Is the diameter normally distributed?
def q6():
    keyword = "yes_or_no"
    stat, p_value_normality = shapiro(df["Diameter"])
    answer = "no" if (p_value_normality < 0.05) else "yes"
    return f"{keyword}={answer}"


# Question 7 - Is there a statistically significant difference in average "Diameter" between the "Sex" categories?
def q7():
    keyword = "yes_or_no"
    anova_stat, anova_p_value = f_oneway(
        df[df["Sex"] == "M"]["Diameter"],
        df[df["Sex"] == "F"]["Diameter"],
        df[df["Sex"] == "I"]["Diameter"],
    )
    answer = "yes" if anova_p_value < 0.05 else "no"
    return f"{keyword}={answer}"


# Question 8 - Create a new variable, "Area", which is the product of "Length" and "Height". Report its median.
def q8():
    global df
    keyword = "median"
    df["Area"] = df["Length"] * df["Height"]
    answer = df["Area"].median()
    return f"{keyword}={answer:.3f}"


# Question 9 - Based on "Area", create a new variable named "LargeArea" with category "Yes" if "Area" is at least the median, "No" otherwise. Find the number of examples with "Yes" for "LargeArea".
def q9():
    global df

    keyword = "n_yes"

    median_area = df["Area"].median()
    df["LargeArea"] = df["Area"].apply(lambda x: "Yes" if x >= median_area else "No")

    answer = df[df["LargeArea"] == "Yes"].shape[0]
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
    return f"{keyword}={test_mae:.3f}"


# Question 11 - Are "LargeArea" and "Sex" statistically independent?
def q11():
    keyword = "yes_or_no"

    # Create a contingency table for 'LargeArea' and 'Sex'
    contingency_table = pd.crosstab(df["LargeArea"], df["Sex"])

    # Perform the chi-squared test for independence
    chi2_stat, p_value_independence, dof, expected = chi2_contingency(contingency_table)

    answer = "no" if p_value_independence < 0.05 else "yes"
    return f"{keyword}={answer}"


def run_abalone():
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
    print(run_abalone())
