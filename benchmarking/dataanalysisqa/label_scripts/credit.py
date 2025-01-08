from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
from sklearn.metrics import r2_score

datasets_dir = Path(__file__).resolve().parent.parent / "datasets"

# import dataset
df = pd.read_csv(datasets_dir / "credit.csv")
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train_idx = df_train.index
df_test_idx = df_test.index
del df_train, df_test


# Question 1 - What is the average income?
def q1():
    keyword = "mean_income"
    answer = df["Income"].mean()
    answer = round(answer, 3)
    return f"{keyword}={answer:.3f}"


# Question 2 - How many are married?
def q2():
    keyword = "n_married"
    answer = df["Married"].value_counts().get("Yes", 0)
    answer = round(answer, 3)
    return f"{keyword}={answer:.3f}"


# Question 3 - What is the average number of cards?
def q3():
    keyword = "mean"
    answer = df["Cards"].mean()
    answer = round(answer, 3)
    return f"{keyword}={answer:.3f}"


# Question 4 - Identify the five highest earners and the five lowest earners. What is the difference between the two groups' average ratings?
def q4():
    keyword = "difference"

    # Identify the five highest and five lowest earners
    highest_earners = df.nlargest(5, "Income")
    lowest_earners = df.nsmallest(5, "Income")

    # Calculate the average ratings for each group
    highest_earners_avg_rating = highest_earners["Rating"].mean()
    lowest_earners_avg_rating = lowest_earners["Rating"].mean()

    # Calculate the difference between the average ratings
    rating_difference = highest_earners_avg_rating - lowest_earners_avg_rating

    answer = round(rating_difference, 3)

    return f"{keyword}={answer:.3f}"


# Question 5 - How many ethnicities are in the dataset?
def q5():
    keyword = "n_ethnicities"
    answer = df["Ethnicity"].nunique()

    answer = round(answer, 3)

    return f"{keyword}={answer:.3f}"


# Question 6 - Make a new variable, "income_categories", based on the income split into "low", "medium", and "high" levels. Define low as < 40. Define medium as at least 40 but less than 80. Define high as at least 80. How many high income earners are there?
def q6():
    global df

    keyword = "n_high"
    df["income_categories"] = df["Income"].apply(
        lambda x: "low" if x < 40 else "medium" if x < 80 else "high"
    )

    # Count the number of high-income earners
    high_income = df[df["income_categories"] == "high"]
    answer = high_income.shape[0]

    answer = round(answer, 3)

    return f"{keyword}={answer:.3f}"


# Question 7 - Does average number of cards differ significantly between the categories of "income_categories"?
def q7():
    keyword = "yes_or_no"

    # Perform a one-way ANOVA test
    _, p_value = stats.f_oneway(
        df.loc[df["income_categories"] == "low", "Cards"],
        df.loc[df["income_categories"] == "medium", "Cards"],
        df.loc[df["income_categories"] == "high", "Cards"],
    )

    # try kruskal test
    pval_kruskal = stats.kruskal(
        df.loc[df["income_categories"] == "low", "Cards"],
        df.loc[df["income_categories"] == "medium", "Cards"],
        df.loc[df["income_categories"] == "high", "Cards"],
    ).pvalue

    assert (p_value <= 0.05) == (pval_kruskal <= 0.05)

    answer = "yes" if p_value <= 0.05 else "no"
    return f"{keyword}={answer}"


# Question 8 - Regress "Limit" on "income_categories" with linear regression. What is the test R-squared?
def q8():
    keyword = "r2"

    df_train = df.loc[df_train_idx]
    df_test = df.loc[df_test_idx]

    # Fit the linear regression model using statsmodels
    model_smf = smf.ols(formula="Limit ~ C(income_categories)", data=df_train).fit()

    # Predict on the test set
    y_pred_smf = model_smf.predict(df_test)

    # Calculate R-squared on the test set
    test_r_squared_smf = r2_score(df_test["Limit"], y_pred_smf)

    test_r_squared_smf = round(test_r_squared_smf, 3)

    return f"{keyword}={test_r_squared_smf:.3f}"


# Question 9 - Regress "Limit" on "income_categories" and "Age" with linear regression. What is the coefficient for "Age"?
def q9():
    keyword = "coef"

    df_train = df.loc[df_train_idx]

    # Fit the model
    model_with_age = smf.ols(
        formula="Limit ~ C(income_categories) + Age", data=df_train
    ).fit()

    # Extract the coefficient for "Age"
    age_coefficient = model_with_age.params["Age"]

    age_coefficient = round(age_coefficient, 3)

    return f"{keyword}={age_coefficient:.3f}"


# Question 10 - Is there a statistically significant difference in means in "Limit" between "Student" levels?
def q10():
    keyword = "yes_or_no"

    student_groups = df.groupby("Student")["Limit"]
    _, p_value = stats.ttest_ind(
        student_groups.get_group("Yes"), student_groups.get_group("No"), equal_var=False
    )
    # test other methods to allow for method flexibility
    pval_student = stats.ttest_ind(
        student_groups.get_group("Yes"), student_groups.get_group("No"), equal_var=True
    ).pvalue

    pval_mannwhitneyu = stats.mannwhitneyu(
        student_groups.get_group("Yes"), student_groups.get_group("No")
    ).pvalue

    assert (p_value <= 0.05) == (pval_mannwhitneyu <= 0.05) == (pval_student <= 0.05)

    answer = "yes" if p_value <= 0.05 else "no"
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
    }


if __name__ == "__main__":
    print(get_labels())
