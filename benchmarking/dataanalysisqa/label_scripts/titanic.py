from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.stats import ttest_ind
import statsmodels.api as sm
from sklearn.metrics import f1_score
import statsmodels.formula.api as smf

datasets_dir = Path(__file__).resolve().parent.parent / "datasets"

# import dataset
df = pd.read_csv(datasets_dir / "titanic.csv")
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train_idx = df_train.index
df_test_idx = df_test.index
del df_train, df_test


# Question 1 - How many passengers survived?
def q1():
    keyword = "n_passengers_survived"
    answer = sum(df["Survived"])
    return f"{keyword}={answer:.3f}"


# Question 2 - How many male and female passengers are there?
def q2():
    keyword1 = "n_male"
    keyword2 = "n_female"
    n_male = sum(df["Sex"] == "male")
    n_female = sum(df["Sex"] == "female")
    return f"{keyword1}={n_male:.3f}, {keyword2}={n_female:.3f}"


# Question 3 - Find the mean, median, and standard deviation of "Age".
def q3():
    keyword1 = "mean"
    keyword2 = "median"
    keyword3 = "std"

    age_mean = df["Age"].mean()
    age_median = df["Age"].median()
    age_std = df["Age"].std()

    return f"{keyword1}={age_mean:.3f}, {keyword2}={age_median:.3f}, {keyword3}={age_std:.3f}"


# Question 4 - How many different values of "Pclass" are there?
def q4():
    keyword = "n_unique"
    num_vals = len(df["Pclass"].unique())
    return f"{keyword}={num_vals:.3f}"


# Question 5 - What's the average "Fare" price?
def q5():
    keyword = "mean"
    ave_fare = df["Fare"].mean()
    return f"{keyword}={ave_fare:.3f}"


# Question 6 - What is the correlation between "Pclass" and "Fare"?
def q6():
    keyword = "corr"
    corr = df["Pclass"].corr(df["Fare"])
    return f"{keyword}={corr:.3f}"


# Question 7 - Is there a statistically significant difference in fare price between those who survived and those who did not?
def q7():
    keyword = "yes_or_no"

    # Condition on survival
    Survived_fare = df[df["Survived"] == 1]["Fare"]
    Dead_fare = df[df["Survived"] == 0]["Fare"]

    # Welch's t-test
    t_stat, p_value = ttest_ind(Survived_fare, Dead_fare, equal_var=False)

    answer = "yes" if (p_value <= 0.05) else "no"
    return f"{keyword}={answer}"


# Question 8 - Is there a statistically significant difference in fare price between men and women?
def q8():
    keyword = "yes_or_no"

    # Condition on Sex
    men_fare = df[df["Sex"] == "male"]["Fare"]
    women_fare = df[df["Sex"] == "female"]["Fare"]

    # Welch's t-test
    t_stat, p_value = ttest_ind(men_fare, women_fare, equal_var=False)

    answer = "yes" if (p_value <= 0.05) else "no"

    return f"{keyword}={answer}"


# Question 9 - Create a new categorical variable, "Age_categorical", with two levels: "young" and "old". Define "old" as those aged at least 50 years. Is there a statistically significant difference in fare price between young and old passengers?
def q9():
    global df
    keyword = "yes_or_no"

    # Create new column
    df["Age_categorical"] = df["Age"].apply(
        lambda x: "old" if x >= 50 else "young" if pd.notnull(x) else None
    )

    young_fare = df.loc[df["Age_categorical"] == "young"]["Fare"]
    old_fare = df.loc[df["Age_categorical"] == "old"]["Fare"]

    # Step 3: Perform a t-test to determine if there is a significant difference
    t_stat, p_value = ttest_ind(young_fare.dropna(), old_fare.dropna())

    answer = "yes" if (p_value <= 0.05) else "no"

    return f"{keyword}={answer}"


# Question 10 - Use logistic regression to predict survival using "Pclass", "Age_categorical", and "Fare". Report the test F1 score.
def q10():
    keyword = "f1"

    df_train = df.loc[df_train_idx]
    df_test = df.loc[df_test_idx]

    df_train = df_train.dropna(subset=["Age_categorical", "Fare", "Pclass", "Survived"])
    df_test = df_test.dropna(subset=["Age_categorical", "Fare", "Pclass", "Survived"])

    logit_model = smf.logit(
        formula="Survived ~ Pclass + Fare + C(Age_categorical)", data=df_train
    ).fit()

    # Make predictions on the test set
    y_test = df_test["Survived"]
    y_pred_proba = logit_model.predict(df_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Calculate the F1 score
    f1 = f1_score(y_test, y_pred)

    return f"{keyword}={f1:.3f}"


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
