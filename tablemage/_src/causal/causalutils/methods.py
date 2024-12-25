import pandas as pd
from typing import Literal
import numpy as np
from joblib import Parallel, delayed
from ...._src.ml.predict.classification import BaseC
from ...._src.data.dataemitter import DataEmitter, PreprocessStepTracer
from ...._src.display.print_utils import print_wrapped


def compute_weights_from_propensity_scores(
    estimand: Literal["ate", "att"],
    propensity_scores: pd.Series,
    treatment: pd.Series,
) -> pd.Series:
    """Computes the weights for the inverse propensity weighting (IPW) estimator.

    Parameters
    ----------
    estimand : Literal["ate", "att"]
        The estimand of interest. 'ate' for Average Treatment Effect,
        'att' for Average Treatment on the Treated.

    propensity_scores : pd.Series
        The propensity scores for the observations.

    treatment : pd.Series
        The treatment indicator variable.

    Returns
    -------
    pd.Series
        The weights for the observations.
    """
    # ensure indices for propensity_scores and treatment are the same
    if not propensity_scores.index.equals(treatment.index):
        raise ValueError(
            "Indices for propensity_scores and treatment must be the same."
        )

    # initialize output series as zeros
    output = pd.Series(0.0, index=propensity_scores.index)

    # Boolean masks for treated and control groups
    idx_for_treatment = treatment == 1
    idx_for_control = treatment == 0

    if estimand == "ate":
        # compute weights for the ATE

        # similar to ifelse(
        # output$treatment == 1, 1 / predictions, 1 / (1 - predictions))
        output.loc[idx_for_treatment] = 1 / propensity_scores.loc[idx_for_treatment]
        output.loc[idx_for_control] = 1 / (1 - propensity_scores.loc[idx_for_control])

    elif estimand == "att":
        # Compute weights for the ATT
        output.loc[idx_for_treatment] = 1  # Weights are 1 for treated individuals

        # Weights for control individuals
        es = propensity_scores.loc[idx_for_control]
        output.loc[idx_for_control] = es / (1 - es)

    else:
        raise ValueError(f"Estimand {estimand} is not supported.")

    return output


def _single_ipw_estimator(
    estimand: Literal["ate", "att"],
    propensity_score_estimator: BaseC,
    X_df: pd.DataFrame,
    Y_series: pd.Series,
    A_series: pd.Series,
) -> float:
    assert len(X_df) == len(Y_series), "Shape mismatch between X and Y"
    assert len(X_df) == len(A_series), "Shape mismatch between X and A"

    if estimand not in ["ate", "att"]:
        raise ValueError(f"Estimand {estimand} is not supported.")

    # Compute e_i, probability of treatment assignment given covariates
    df = X_df.join(A_series)
    propensity_score_estimator.specify_data(
        dataemitter=DataEmitter(
            df_train=df,
            df_test=df,
            y_var=A_series.name,
            X_vars=X_df.columns.to_list(),
            step_tracer=PreprocessStepTracer(),
        )
    )
    propensity_score_estimator.fit()
    e_numpy = propensity_score_estimator._train_scorer._y_pred_score
    Y_numpy = Y_series.to_numpy()
    A_numpy = A_series.to_numpy()

    assert len(e_numpy) == len(Y_numpy), "Shape mismatch between e and Y"
    assert len(e_numpy) == len(A_numpy), "Shape mismatch between e and A"

    if estimand == "ate":
        estimate = (Y_numpy * A_numpy / e_numpy).mean() / (A_numpy / e_numpy).mean() - (
            Y_numpy * (1 - A_numpy) / (1 - e_numpy)
        ).mean() / ((1 - A_numpy) / (1 - e_numpy)).mean()
    else:
        estimate = (Y_numpy * A_numpy).mean() / A_numpy.mean() - (
            Y_numpy * (1 - A_numpy) * e_numpy / (1 - e_numpy)
        ).mean() / ((1 - A_numpy) * e_numpy / (1 - e_numpy)).mean()

    return estimate


def compute_bootstrapped_ipw_estimator(
    estimand: Literal["ate", "att"],
    propensity_score_estimator: BaseC,
    n_bootstraps: int,
    X_df: pd.DataFrame,
    Y_series: pd.Series,
    A_series: pd.Series,
) -> tuple[float, float]:
    """Estimates the average treatment effect (ATE) or ATT using the IPW estimator with bootstrapping.

    Parameters
    ----------
    estimand : Literal["ate", "att"]
        The estimand of interest. 'ate' for Average Treatment Effect,
        'att' for Average Treatment on the Treated.
    propensity_score_estimator : BaseC
        The propensity score estimator to use. Must have `fit` and `predict` methods.
    n_bootstraps : int
        The number of bootstrap samples to use.
    X_df : pd.DataFrame
        Covariates used to estimate propensity scores.
    Y_series : pd.Series
        Outcome variable.
    A_series : pd.Series
        Treatment assignment indicator (0 or 1).

    Returns
    -------
    tuple[float, float]
        A tuple containing the estimated effect and its bootstrap standard error.
    """
    effect = _single_ipw_estimator(
        estimand=estimand,
        propensity_score_estimator=propensity_score_estimator,
        X_df=X_df,
        Y_series=Y_series,
        A_series=A_series,
    )

    X_df_temp = X_df.reset_index(drop=True)
    Y_series_temp = Y_series.reset_index(drop=True)
    A_series_temp = A_series.reset_index(drop=True)

    def bootstrap_sample(idx, seed):
        idx = idx + 1
        if idx % 20 == 0:
            print_wrapped(
                f"IPW estimator bootstrap sample {idx}/{n_bootstraps}", type="UPDATE"
            )
        np.random.seed(seed)
        idx = np.random.choice(X_df_temp.index, size=len(X_df_temp), replace=True)
        X_indexed = X_df_temp.loc[idx].reset_index(drop=True)
        Y_indexed = Y_series_temp.loc[idx].reset_index(drop=True)
        A_indexed = A_series_temp.loc[idx].reset_index(drop=True)
        return _single_ipw_estimator(
            estimand=estimand,
            propensity_score_estimator=propensity_score_estimator,
            X_df=X_indexed,
            Y_series=Y_indexed,
            A_series=A_indexed,
        )

    seeds = np.random.randint(0, 1000000, size=n_bootstraps)
    bootstrapped_effects = Parallel(n_jobs=-1)(
        delayed(bootstrap_sample)(idx, seed) for idx, seed in enumerate(seeds)
    )
    bootstrap_std_error = np.std(bootstrapped_effects, ddof=1)
    return effect, bootstrap_std_error
