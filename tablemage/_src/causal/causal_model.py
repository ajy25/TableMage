import pandas as pd
from typing import Literal
import numpy as np


from .report import CausalReport
from .causalutils.methods import (
    compute_weights_from_propensity_scores,
    compute_bootstrapped_ipw_estimator,
)
from .causalutils.models import SimpleOLS

from ..._src.ml.predict.classification import BaseC, LinearC
from ..._src.data.datahandler import DataHandler


class CausalModel:
    """A class for estimating causal effects."""

    def __init__(
        self,
        datahandler: DataHandler,
        treatment: str,
        outcome: str,
        confounders: list[str],
        dataset: Literal["train", "all"] = "all",
    ):
        """Initializes a CausalModel object. Strictly handles simple
        causal inference tasks, that is where we have the following causal DAG:
        (
        A -> Y,
        X -> Y,
        X -> A,
        U -> Y,
        U -> A
        )
        where A is the treatment variable, Y is the outcome variable, X is the
        confounding variables, and U is the unobserved confounding variables.
        Only estimates either ATE or ATT.

        Parameters
        ----------
        datahandler: DataHandler
            The DataHandler object. Both the training and test data will be used
            for causal inference (i.e. we will concatenate the training and test
            data).

        treatment : str
            The name of the treatment variable.

        outcome : str
            The name of the outcome variable.

        confounders : list[str]
            The names of the confounding variables.

        dataset : Literal["train", "all"]
            The dataset to use for causal inference. "train" for training data,
            "test" for test data, and "all" for both training and test data,
            by default "all". If the Analyzer object was not
            initialized with a test set, use "train" to
            obtain all the data, since in this case the test set would be set
            to a copy of train.
        """
        temp_datahandler = datahandler.copy()
        temp_datahandler._verbose = False
        temp_datahandler.dropna(include_vars=[treatment, outcome] + confounders)
        self._treatment = treatment
        self._outcome = outcome
        self._confounders = confounders
        if dataset == "train":
            self._emitter = temp_datahandler.train_test_emitter(
                y_var=outcome,
                X_vars=confounders + [treatment],
            )
            self._propensity_emitter = temp_datahandler.train_test_emitter(
                y_var=treatment,
                X_vars=confounders,
            )
            self._X_df, self._y_series = self._emitter.emit_train_Xy()
        elif dataset == "all":
            self._emitter = temp_datahandler.full_dataset_emitter(
                y_var=outcome,
                X_vars=confounders + [treatment],
            )
            self._propensity_emitter = temp_datahandler.full_dataset_emitter(
                y_var=treatment,
                X_vars=confounders,
            )
            self._X_df, self._y_series = self._emitter.emit_train_Xy()
        else:
            raise ValueError("Invalid dataset.")

        if len(self._X_df) == 0:
            raise ValueError("No data left after dropping missing values.")
        if treatment not in self._X_df.columns:
            raise ValueError("Treatment variable not found in data.")
        if self._X_df[treatment].nunique() != 2:
            raise ValueError("Treatment variable must be binary.")
        if not self._X_df[treatment].isin([0, 1]).all():
            raise ValueError(
                "Treatment variable must be binary numeric (0 or 1 valued)."
            )

    def estimate_ate(
        self,
        method: Literal[
            "naive", "outcome_regression", "ipw_weighted_regression", "ipw_estimator"
        ],
        propensity_score_estimator: BaseC = LinearC(
            type="no_penalty",
        ),
        robust_se: Literal["nonrobust", "HC0", "HC1", "HC2", "HC3"] = "nonrobust",
        n_bootstraps: int = 100,
    ) -> CausalReport:
        """Estimates the average treatment effect (ATE).

        Parameters
        ----------
        method : Literal["outcome_regression", "ipw_weighted_regression",
        "ipw_estimator"]
            The method for estimating the ATE.

        propensity_score_estimator : BaseC, optional
            The estimator/model for computing/predicting the propensity scores,
            by default LinearC(type="no_penalty") (logistic regression).
            Hyperparameters will be selected as specified in the BaseC model.

        robust_se : Literal["nonrobust", "HC0", "HC1", "HC2", "HC3"], optional
            The type of robust standard errors to use, by default "nonrobust".
            If "nonrobust", then the standard errors are not robust.

        n_bootstraps : int, optional
            The number of bootstraps to use for the IPW estimator, by default 100.
            Ignored if method is not "ipw_estimator".
        """
        if method == "naive":
            return self._naive()
        elif method == "outcome_regression":
            return self._outcome_regression(robust_se=robust_se)
        elif method == "ipw_weighted_regression":
            return self._ipw_weighted_regression(
                estimand="ate",
                propensity_score_estimator=propensity_score_estimator,
                robust_se=robust_se,
            )
        elif method == "ipw_estimator":
            return self._ipw_estimator(
                estimand="ate",
                propensity_score_estimator=propensity_score_estimator,
                n_bootstraps=n_bootstraps,
            )
        else:
            raise ValueError("Invalid method.")

    def estimate_att(
        self,
        method: Literal["ipw_weighted_regression", "ipw_estimator"],
        propensity_score_estimator: BaseC = LinearC(
            type="no_penalty",
        ),
        robust_se: Literal["nonrobust", "HC0", "HC1", "HC2", "HC3"] = "nonrobust",
        n_bootstraps: int = 100,
    ) -> CausalReport:
        """Estimates the average treatment effect on the treated (ATT).

        Parameters
        ----------
        method : Literal["ipw_weighted_regression", "ipw_estimator"]
            The method for estimating the ATT.

        propensity_score_estimator : BaseC, optional
            The estimator/model for computing/predicting the propensity scores,
            by default LinearC(type="no_penalty") (logistic regression).
            Hyperparameters will be selected as specified in the BaseC model.

        robust_se : Literal["nonrobust", "HC0", "HC1", "HC2", "HC3"], optional
            The type of robust standard errors to use, by default "nonrobust".
            If "nonrobust", then the standard errors are not robust.

        n_bootstraps : int, optional
            The number of bootstraps to use for the IPW estimator, by default 100.
            Ignored if method is not "ipw_estimator".
        """
        if method == "ipw_weighted_regression":
            return self._ipw_weighted_regression(
                estimand="att",
                propensity_score_estimator=propensity_score_estimator,
                robust_se=robust_se,
            )
        elif method == "ipw_estimator":
            return self._ipw_estimator(
                estimand="att",
                propensity_score_estimator=propensity_score_estimator,
                n_bootstraps=n_bootstraps,
            )
        else:
            raise ValueError("Invalid method.")

    def _naive(self) -> CausalReport:
        "Computes the difference in means between the treated and untreated."
        method_description = (
            "Naive estimator. "
            "Computes the difference in means between the treated and untreated. "
            "No adjustment for confounders."
        )
        X_df, y_df = self._X_df, self._y_series
        treated = y_df[X_df[self._treatment] == 1]
        untreated = y_df[X_df[self._treatment] == 0]
        estimate = treated.mean() - untreated.mean()
        se = np.sqrt(treated.var() / len(treated) + untreated.var() / len(untreated))

        return CausalReport(
            estimate=estimate,
            se=se,
            n_units=len(X_df),
            n_units_treated=len(X_df[X_df[self._treatment] == 1]),
            outcome_var=self._outcome,
            treatment_var=self._treatment,
            confounders=self._confounders,
            estimand="ate",
            method="Naive Estimator (Difference in Means)",
            method_description=method_description,
        )

    def _outcome_regression(self, robust_se: str = "nonrobust") -> CausalReport:
        """Outcome regression with OLS. Confounders are included as predictors.
        Only relevant for estimating the ATE.
        """
        method_description = (
            "Outcome regression with OLS. "
            "Fits a linear regression model to the outcome variable using the "
            "treatment variable and confounders as predictors."
        )
        df_X, df_y = self._X_df, self._y_series

        print(df_X.columns.to_list())

        model = SimpleOLS(y=df_y, X=df_X, robust=robust_se)

        estimate = model.get_coef(self._treatment)
        se = model.get_se(self._treatment)
        p_val = model.get_pvalue(self._treatment)

        return CausalReport(
            estimate=estimate,
            se=se,
            n_units=len(df_X),
            n_units_treated=len(df_X[df_X[self._treatment] == 1]),
            outcome_var=self._outcome,
            treatment_var=self._treatment,
            confounders=self._confounders,
            estimand="ate",
            method="Outcome Regression",
            method_description=method_description,
            p_value=p_val,
        )

    def _ipw_weighted_regression(
        self,
        estimand: Literal["ate", "att"],
        propensity_score_estimator: BaseC = LinearC(
            type="no_penalty",
        ),
        robust_se: str = "nonrobust",
    ) -> CausalReport:
        """Inverse probability weighting (IPW)-weighted regression (WLS).

        Parameters
        ----------
        estimand : Literal["ate", "att"]
            The estimand. "ate" for average treatment effect and "att" for average
            treatment effect on the treated.

        propensity_score_estimator : BaseC, optional
            The estimator/model for computing/predicting the propensity scores,
            by default LinearC(type="no_penalty") (logistic regression).
            Hyperparameters will be selected as specified in the model.
        """
        method_description = (
            "Inverse probability weighting (IPW)-weighted regression (WLS). "
            "Weights units via the inverse of their propensity scores. "
            "Uses logistic regression to estimate the propensity scores."
        )

        emitter = self._emitter

        X_df, y_series = emitter.emit_train_Xy()
        treatment_series = X_df[self._treatment]

        propensity_emitter = self._propensity_emitter
        propensity_score_estimator.specify_data(dataemitter=propensity_emitter)
        propensity_score_estimator.fit()

        propensity_scores_series = pd.Series(
            propensity_score_estimator._train_scorer._y_pred_score, index=X_df.index
        )

        weights = compute_weights_from_propensity_scores(
            propensity_scores=propensity_scores_series,
            treatment=treatment_series,
            estimand=estimand,
        )

        model = SimpleOLS(y=y_series, X=X_df, weights=weights, robust=robust_se)

        estimate = model.get_coef(self._treatment)
        se = model.get_se(self._treatment)
        p_val = model.get_pvalue(self._treatment)

        return CausalReport(
            estimate=estimate,
            se=se,
            n_units=len(X_df),
            n_units_treated=len(X_df[X_df[self._treatment] == 1]),
            outcome_var=self._outcome,
            treatment_var=self._treatment,
            confounders=self._confounders,
            estimand=estimand,
            method="Inverse Probability Weighting (IPW) Weighted Regression",
            method_description=method_description,
            p_value=p_val,
        )

    def _ipw_estimator(
        self,
        estimand: Literal["ate", "att"],
        propensity_score_estimator: BaseC = LinearC(
            type="no_penalty",
        ),
        n_bootstraps: int = 1000,
    ) -> CausalReport:
        """Estimates the average treatment effect (ATE) using the IPW estimator.

        Parameters
        ----------
        estimand : Literal["ate", "att"]
            The estimand of interest. 'ate' for Average Treatment Effect,
            'att' for Average Treatment on the Treated.

        propensity_score_estimator : BaseC
            The estimator/model for computing/predicting the propensity scores.

        n_bootstraps : int, optional
        """
        method_description = "Inverse probability weighting estimator."
        emitter = self._emitter

        X_df, y_series = emitter.emit_train_Xy()
        treatment_series = X_df[self._treatment].copy()
        X_df_no_trmt = X_df.drop(columns=[self._treatment])

        estimate, se = compute_bootstrapped_ipw_estimator(
            estimand=estimand,
            propensity_score_estimator=propensity_score_estimator,
            n_bootstraps=n_bootstraps,
            X_df=X_df_no_trmt,
            Y_series=y_series,
            A_series=treatment_series,
        )

        return CausalReport(
            estimate=estimate,
            se=se,
            n_units=len(X_df),
            n_units_treated=len(X_df[X_df[self._treatment] == 1]),
            outcome_var=self._outcome,
            treatment_var=self._treatment,
            confounders=self._confounders,
            estimand=estimand,
            method="Inverse Probability Weighting (IPW) Estimator",
            method_description=method_description,
        )
