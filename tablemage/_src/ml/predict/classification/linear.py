from sklearn.linear_model import LogisticRegression
from typing import Mapping, Literal, Iterable


from .base import BaseC, HyperparameterSearcher
from optuna.distributions import (
    FloatDistribution,
    BaseDistribution,
)
from ....feature_selection import BaseFSC


class LinearC(BaseC):
    """Logistic regression classifier.

    Hyperparameter optimization is performed automatically during training.
    The hyperparameter search process can be modified by the user.
    """

    def __init__(
        self,
        type: Literal["no_penalty", "l1", "l2", "elasticnet"] = "l2",
        hyperparam_search_method: Literal["optuna", "grid"] | None = None,
        hyperparam_search_space: (
            Mapping[str, Iterable | BaseDistribution] | None
        ) = None,
        feature_selectors: list[BaseFSC] | None = None,
        max_n_features: int | None = None,
        model_random_state: int = 42,
        name: str | None = None,
        threshold_strategy: Literal["f1", "roc"] | None = "roc",
        **kwargs,
    ):
        """
        Initializes a LinearC object.

        Parameters
        ----------
        type: Literal['no_penalty', 'l1', 'l2', 'elasticnet']
            Default: 'l2'. Specifies the type of logistic regression penalty.

        hyperparam_search_method : Literal[None, 'grid', 'optuna']
            Default: None. If None, a model-specific default hyperparameter search
            is conducted.

        hyperparam_search_space : Mapping[str, Iterable | BaseDistribution]
            Default: None. If None, a model-specific default hyperparameter search
            is conducted.

        feature_selectors : list[BaseFSC]
            Default: None. If not None, specifies the feature selectors for the
            VotingSelectionReport.

        max_n_features : int | None
            Default: None.
            Only useful if feature_selectors is not None.
            If None, then all features with at least 50% support are selected.

        model_random_state : int
            Default: 42. Random seed for the model.

        name : str
            Default: None. Determines how the model shows up in the reports.
            If None, a default name is set based on the type of the model.

        threshold_strategy : Literal['f1', 'roc'] | None
            Default: 'f1'. Determines the decision threshold optimization strategy.
            'f1' uses the F1 score, 'roc' uses the ROC curve.
            If None, no threshold optimization is performed.
            Only considered if model yields probabilities.

        **kwargs : dict
            Key word arguments are passed directly into the intialization of the
            HyperparameterSearcher class. See below for options.

            inner_cv : int | BaseCrossValidator
                Default: 5. Number of inner cross validation folds. Inner
                cross validation is used for hyperparameter optimization.

            inner_cv_seed : int
                Default: 42. Random seed for inner cross validation.

            n_jobs : int
                Default: 1. Number of parallel jobs to run.

            verbose : int
                Default: 0. Sets the sklearn verbosity level for the sklearn estimator.
                2 is the most verbose.

            n_trials : int
                Default: 100. Number of trials for hyperparameter optimization. Only
                used if hyperparam_search_method is 'optuna'.
        """
        super().__init__(threshold_strategy=threshold_strategy)
        self._dropfirst = True
        self._feature_selectors = feature_selectors
        self._max_n_features = max_n_features
        if name is None:
            self._name = f"LinearC({type})"
        else:
            self._name = name

        if type == "no_penalty":
            self._best_estimator = LogisticRegression(
                penalty=None, random_state=model_random_state, max_iter=1000
            )
            if (hyperparam_search_method is None) or (hyperparam_search_space is None):
                hyperparam_search_method = None
                hyperparam_search_space = None

        elif type == "l1":
            self._best_estimator = LogisticRegression(
                penalty="l1", random_state=model_random_state, solver="liblinear"
            )
            if (hyperparam_search_method is None) or (hyperparam_search_space is None):
                hyperparam_search_method = "optuna"
                hyperparam_search_space = {"C": FloatDistribution(1e-2, 1e2, log=True)}

        elif type == "l2":
            self._best_estimator = LogisticRegression(
                penalty="l2", random_state=model_random_state
            )
            if (hyperparam_search_method is None) or (hyperparam_search_space is None):
                hyperparam_search_method = "optuna"
                hyperparam_search_space = {"C": FloatDistribution(1e-2, 1e2, log=True)}

        elif type == "elasticnet":
            self._best_estimator = LogisticRegression(
                penalty="elasticnet", random_state=model_random_state, solver="saga"
            )
            if (hyperparam_search_method is None) or (hyperparam_search_space is None):
                hyperparam_search_method = "optuna"
                hyperparam_search_space = {
                    "C": FloatDistribution(1e-2, 1e2, log=True),
                    "l1_ratio": FloatDistribution(0.0, 1.0),
                }
        else:
            raise ValueError("Invalid value for type")

        self._hyperparam_searcher = HyperparameterSearcher(
            estimator=self._best_estimator,
            method=hyperparam_search_method,
            hyperparam_grid=hyperparam_search_space,
            estimator_name=self._name,
            **kwargs,
        )

        self._validate_inputs()
