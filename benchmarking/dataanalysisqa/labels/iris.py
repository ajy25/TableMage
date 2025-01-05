from pathlib import Path
import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import train_test_split

datasets_dir = Path(__file__).resolve().parent.parent / "datasets"

# import dataset
df = pd.read_csv(datasets_dir / "iris.csv")
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)


# Question 1 - Compute the mean and median of "SepalLengthCm".
def q1():
    keyword1 = "mean"
    keyword2 = "median"
    answer1 = df["SepalLengthCm"].mean()
    answer2 = df["SepalLengthCm"].median()
    return f"{keyword1}={answer1:.3f}, {keyword2}={answer2:.3f}"


# Question 2 - Is the distribution of "SepalLengthCm" normal?
def q2():
    keyword = "yes_or_no"
    _, pval = stats.normaltest(df["SepalLengthCm"])
    return f"{keyword}={'yes' if pval > 0.05 else 'no'}"


# Question 3 - How many different species categories are there?
def q3():
    keyword = "n_species"
    answer = df["Species"].nunique()
    return f"{keyword}={answer:.3f}"


# Question 4 - What is the mean "SepalLengthCm" for species "Iris-setosa"?
def q4():
    keyword = "mean"
    answer = df[df["Species"] == "Iris-setosa"]["SepalLengthCm"].mean()
    return f"{keyword}={answer:.3f}"


# Question 5 - Find the correlation between "PetalWidthCm" and "PetalLengthCm". What is the correlation coefficient, and what is the p-value?
def q5():
    keyword1 = "corr"
    keyword2 = "pval"
    corr, pval = stats.pearsonr(df["PetalWidthCm"], df["PetalLengthCm"])
    return f"{keyword1}={corr:.3f}, {keyword2}={pval:.3f}"


# Question 6 - Make a new variable named "PetalAreaCm" that is defined as the product of "PetalWidthCm" and "PetalLengthCm". What is the mean and standard deviation of this new variable?
def q6():
    global df, df_train, df_test
    keyword1 = "mean"
    keyword2 = "std"
    df["PetalAreaCm"] = df["PetalWidthCm"] * df["PetalLengthCm"]
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    answer1 = df["PetalAreaCm"].mean()
    answer2 = df["PetalAreaCm"].std()
    return f"{keyword1}={answer1:.3f}, {keyword2}={answer2:.3f}"


# Question 7 - Find the mean "PetalAreaCm" for species "Iris-setosa".
def q7():
    keyword = "mean"
    answer = df[df["Species"] == "Iris-setosa"]["PetalAreaCm"].mean()
    return f"{keyword}={answer:.3f}"


# Question 8 - Is there a statistically significant correlation between "SepalLengthCm" and "PetalAreaCm"?
def q8():
    keyword = "yes_or_no"
    corr, pval = stats.pearsonr(df["SepalLengthCm"], df["PetalAreaCm"])
    return f"{keyword}={'yes' if pval < 0.05 else 'no'}"


# Question 9 - Engineer a new variable, "LargeArea", that is given label "large" if "PetalAreaCm" is at least its median and label "small" if "PetalAreaCm" is less than its median. Report the number of "large" observations.
def q9():
    keyword = "n_large"
    df["LargeArea"] = df["PetalAreaCm"].apply(
        lambda x: "large" if x >= df["PetalAreaCm"].median() else "small"
    )
    answer = df[df["LargeArea"] == "large"].shape[0]
    return f"{keyword}={answer:.3f}"


def run_iris():
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
    print(run_iris())
