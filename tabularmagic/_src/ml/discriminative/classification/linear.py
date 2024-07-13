import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Mapping, Literal, Iterable


from .base import BaseC, HyperparameterSearcher
from optuna.distributions import (
    FloatDistribution,
    CategoricalDistribution,
    BaseDistribution,
)


class LinearC(BaseC):
    """Logistic Regression classifier.

    Like all BaseC-derived classes, hyperparameter selection is
    performed automatically during training. The cross-validation and
    hyperparameter selection process can be modified by the user.
    """

    def __init__(
        self,
        type: Literal["no_penalty", "l1", "l2", "elasticnet"] = "l2",
        hyperparam_search_method: Literal["optuna", "grid"] | None = None,
        hyperparam_grid_specification: (
            Mapping[str, Iterable | BaseDistribution] | None
        ) = None,
        model_random_state: int = 42,
        name: str | None = None,
        **kwargs,
    ):
        """
        Initializes a LinearC object.

        Parameters
        ----------
        type: Literal['no_penalty', 'l1', 'l2', 'elasticnet'].
            Default: 'l2'.
        hyperparam_search_method : Literal[None, 'grid', 'optuna'].
            Default: None. If None, a classification-specific default hyperparameter
            search is conducted.
        hyperparam_grid_specification : Mapping[str, Iterable | BaseDistribution].
            Default: None. If None, a classification-specific default hyperparameter
            search is conducted.
        model_random_state : int.
            Default: 42. Random seed for the model.
        name : str.
            Default: None. Determines how the model shows up in the reports.
            If None, the name is set to be the class name.
        kwargs : Key word arguments are passed directly into the
            intialization of the HyperparameterSearcher class. In particular,
            inner_cv and inner_cv_seed can be set via kwargs.

        **kwargs
        --------------
        inner_cv : int | BaseCrossValidator.
            Default: 5.
        inner_cv_seed : int.
            Default: 42.
        n_jobs : int.
            Default: 1. Number of parallel jobs to run.
        verbose : int.
            Default: 0. scikit-learn verbosity level.
        n_trials : int.
            Default: 100. Number of trials for hyperparameter optimization. Only
            used if hyperparam_search_method is 'optuna'.
        """
        super().__init__()
        self._dropfirst = True

        if name is None:
            self._name = f"LinearC({type})"
        else:
            self._name = name

        if type == "no_penalty":
            self._estimator = LogisticRegression(
                penalty=None, random_state=model_random_state
            )
            if (hyperparam_search_method is None) or (
                hyperparam_grid_specification is None
            ):
                hyperparam_search_method = "grid"
                hyperparam_grid_specification = {"fit_intercept": [True]}
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self._estimator,
                method=hyperparam_search_method,
                hyperparam_grid=hyperparam_grid_specification,
                **kwargs,
            )

        elif type == "l1":
            self._estimator = LogisticRegression(
                penalty="l1", random_state=model_random_state, solver="liblinear"
            )
            if (hyperparam_search_method is None) or (
                hyperparam_grid_specification is None
            ):
                hyperparam_search_method = "optuna"
                hyperparam_grid_specification = {
                    "C": FloatDistribution(1e-2, 1e2, log=True)
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self._estimator,
                method=hyperparam_search_method,
                hyperparam_grid=hyperparam_grid_specification,
                **kwargs,
            )

        elif type == "l2":
            self._estimator = LogisticRegression(
                penalty="l2", random_state=model_random_state
            )
            if (hyperparam_search_method is None) or (
                hyperparam_grid_specification is None
            ):
                hyperparam_search_method = "optuna"
                hyperparam_grid_specification = {
                    "C": FloatDistribution(1e-2, 1e2, log=True)
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self._estimator,
                method=hyperparam_search_method,
                hyperparam_grid=hyperparam_grid_specification,
                **kwargs,
            )

        elif type == "elasticnet":
            self._estimator = LogisticRegression(
                penalty="elasticnet", random_state=model_random_state, solver="saga"
            )
            if (hyperparam_search_method is None) or (
                hyperparam_grid_specification is None
            ):
                hyperparam_search_method = "optuna"
                hyperparam_grid_specification = {
                    "C": FloatDistribution(1e-2, 1e2, log=True),
                    "l1_ratio": FloatDistribution(0.0, 1.0),
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self._estimator,
                method=hyperparam_search_method,
                hyperparam_grid=hyperparam_grid_specification,
                **kwargs,
            )

        else:
            raise ValueError("Invalid value for type")
