from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.preprocessing import MinMaxScaler
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler

datasets_dir = Path(__file__).resolve().parent.parent / "datasets"

# import dataset
df = pd.read_csv(datasets_dir / "breast_cancer.csv")
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train_idx = df_train.index
df_test_idx = df_test.index
del df_train, df_test


# Question 1 - Find the average and standard deviation of the mean radius.
def q1():
    keyword1 = "mean"
    keyword2 = "std"

    mean_radius_avg = df["mean radius"].mean()
    mean_radius_std = df["mean radius"].std()

    mean, std = mean_radius_avg, mean_radius_std
    return f"{keyword1}={mean:.3f}, {keyword2}={std:.3f}"


# Question 2 - Compute the correlation between mean radius and the breast cancer indicator variable.
def q2():
    keyword = "corr"
    correlation = df["mean radius"].corr(df["breast_cancer_yn"])
    return f"{keyword}={correlation:.3f}"


# Question 3 - Is there a difference in mean radius between those with and those without breast cancer?
def q3():
    keyword = "yes_or_no"
    group_with_cancer = df[df["breast_cancer_yn"] == 1]["mean radius"]
    group_without_cancer = df[df["breast_cancer_yn"] == 0]["mean radius"]

    # Perform a t-test to compare the means of the two groups
    t_stat, p_value = ttest_ind(
        group_with_cancer, group_without_cancer, equal_var=False
    )  # Welch's t-test

    answer = "yes" if p_value < 0.05 else "no"
    return f"{keyword}={answer}"


# Question 4 - Is there a difference in area error between those with and those without breast cancer?
def q4():
    keyword = "yes_or_no"
    # Separate the "area error" data for both groups
    area_error_with_cancer = df[df["breast_cancer_yn"] == 1]["area error"]
    area_error_without_cancer = df[df["breast_cancer_yn"] == 0]["area error"]

    # Perform a t-test to compare the means of the "area error" for both groups using scipy
    t_stat_area, p_value_area = ttest_ind(
        area_error_with_cancer, area_error_without_cancer, equal_var=False
    )  # Welch's t-test

    answer = "yes" if p_value_area < 0.05 else "no"
    return f"{keyword}={answer}"


# Question 5 - Min-max scale mean radius. Then, regress with linear regression the breast cancer indicator on mean radius, and report the coefficient for mean radius.
def q5():
    global df

    keyword = "coef"
    df_train = df.loc[df_train_idx]
    df_test = df.loc[df_test_idx]

    # Min-max scale "mean radius" in the training set
    scaler_minmax = MinMaxScaler()
    df_train["mean radius"] = scaler_minmax.fit_transform(df_train[["mean radius"]])
    df_test["mean radius"] = scaler_minmax.transform(df_test[["mean radius"]])

    df.loc[df_train_idx, "mean radius"] = df_train["mean radius"]
    df.loc[df_test_idx, "mean radius"] = df_test["mean radius"]

    # Perform the regression using statsmodels for "mean radius" (training set)
    minmax_model_train = smf.ols(
        'breast_cancer_yn ~ Q("mean radius")', data=df_train
    ).fit()

    mean_radius_scaled_coef_train = minmax_model_train.params['Q("mean radius")']
    answer = mean_radius_scaled_coef_train
    return f"{keyword}={answer:.3f}"


# Question 6 - Standard scale mean area. Then, regress with linear regression the breast cancer indicator on mean area, and report the coefficient for mean area.
def q6():
    global df

    keyword = "coef"
    df_train = df.loc[df_train_idx]
    df_test = df.loc[df_test_idx]

    # Standard scale "mean area" in the training set
    scaler_standard = StandardScaler()
    df_train["mean area"] = scaler_standard.fit_transform(df_train[["mean area"]])
    df_test["mean area"] = scaler_standard.transform(df_test[["mean area"]])

    df.loc[df_train_idx, "mean area"] = df_train["mean area"]
    df.loc[df_test_idx, "mean area"] = df_test["mean area"]

    # Perform the regression using statsmodels for "mean area" (training set)
    standard_model_train = smf.ols(
        'breast_cancer_yn ~ Q("mean area")', data=df_train
    ).fit()

    # Print the coefficient and summary
    mean_area_scaled_coef_train = standard_model_train.params['Q("mean area")']
    answer = mean_area_scaled_coef_train
    return f"{keyword}={answer:.3f}"


# Question 7 - Find the difference in the mean area between those with and those without breast cancer.
def q7():
    keyword = "difference"

    # Calculate the mean area for both groups: with and without breast cancer
    mean_area_with_cancer = df[df["breast_cancer_yn"] == 1]["mean area"].mean()
    mean_area_without_cancer = df[df["breast_cancer_yn"] == 0]["mean area"].mean()

    # Calculate the difference
    difference_in_mean_area = mean_area_without_cancer - mean_area_with_cancer

    answer = difference_in_mean_area
    return f"{keyword}={answer:.3f}"


# Question 8 - What is the fifth largest mean radius value?
def q8():
    keyword = "value"
    fifth_largest_mean_radius = df["mean radius"].nlargest(5).iloc[-1]

    return f"{keyword}={fifth_largest_mean_radius:.3f}"


# Question 9 - Compute the interquartile range of "mean radius". Identify outliers, based on 1.5 times the interquartile range. How many outliers are there?
def q9():
    keyword = "n_outliers"

    # Calculate the interquartile range (IQR) for "mean radius"
    Quartile1 = df["mean radius"].quantile(0.25)
    Quartile3 = df["mean radius"].quantile(0.75)
    IQR = Quartile3 - Quartile1

    # Determine the bounds for outliers
    lower_bound = Quartile1 - 1.5 * IQR
    upper_bound = Quartile3 + 1.5 * IQR

    # Identify outliers
    outliers = df[(df["mean radius"] < lower_bound) | (df["mean radius"] > upper_bound)]

    # Count the number of outliers
    num_outliers = len(outliers)
    return f"{keyword}={num_outliers:.3f}"


def run_breast_cancer():
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
    print(run_breast_cancer())
