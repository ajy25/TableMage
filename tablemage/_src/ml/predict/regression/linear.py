from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    HuberRegressor,
    RANSACRegressor,
)
from typing import Mapping, Literal, Iterable
from .base import BaseR, HyperparameterSearcher
from ....feature_selection import BaseFSR
from optuna.distributions import (
    FloatDistribution,
    CategoricalDistribution,
    BaseDistribution,
)


class LinearR(BaseR):
    """Linear regression.

    Hyperparameter optimization is performed automatically during training.
    The hyperparameter search process can be modified by the user.
    """

    def __init__(
        self,
        type: Literal["ols", "l1", "l2", "elasticnet"] = "ols",
        hyperparam_search_method: Literal["optuna", "grid"] | None = None,
        hyperparam_search_space: (
            Mapping[str, Iterable | BaseDistribution] | None
        ) = None,
        feature_selectors: list[BaseFSR] | None = None,
        max_n_features: int | None = None,
        model_random_state: int = 42,
        name: str | None = None,
        **kwargs,
    ):
        """
        Initializes a LinearR object.

        Parameters
        ----------
        type : Literal['ols', 'l1', 'l2', 'elasticnet']
            Default: 'ols'. The type of linear regression to be used.

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
            If None, the name is set to be the class name.

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
        super().__init__()
        self._dropfirst = True  # we want to drop first for linear models
        self._feature_selectors = feature_selectors
        self._max_n_features = max_n_features
        self._type = type
        if name is None:
            self._name = f"LinearR({self._type})"
        else:
            self._name = name

        if type == "ols":
            self._best_estimator = LinearRegression()
            if (hyperparam_search_method is None) or (hyperparam_search_space is None):
                hyperparam_search_method = None
                hyperparam_search_space = None
        elif type == "l1":
            self._best_estimator = Lasso(
                selection="random", random_state=model_random_state
            )
            if (hyperparam_search_method is None) or (hyperparam_search_space is None):
                hyperparam_search_method = "optuna"
                hyperparam_search_space = {
                    "alpha": FloatDistribution(1e-5, 1e1, log=True)
                }
        elif type == "l2":
            self._best_estimator = Ridge(random_state=model_random_state)
            if (hyperparam_search_method is None) or (hyperparam_search_space is None):
                hyperparam_search_method = "optuna"
                hyperparam_search_space = {
                    "alpha": FloatDistribution(1e-5, 1e1, log=True)
                }
        elif type == "elasticnet":
            self._best_estimator = ElasticNet(
                selection="random", random_state=model_random_state
            )
            if (hyperparam_search_method is None) or (hyperparam_search_space is None):
                hyperparam_search_method = "optuna"
                hyperparam_search_space = {
                    "alpha": FloatDistribution(1e-5, 1e1, log=True),
                    "l1_ratio": FloatDistribution(0.0, 1.0),
                }
        else:
            raise ValueError(f"Invalid value for type: {type}.")

        self._hyperparam_searcher = HyperparameterSearcher(
            estimator=self._best_estimator,
            method=hyperparam_search_method,
            hyperparam_grid=hyperparam_search_space,
            estimator_name=self._name,
            **kwargs,
        )

        self._validate_inputs()


class RobustLinearR(BaseR):
    """Robust linear regressor.

    Hyperparameter optimization is performed automatically during training.
    The hyperparameter search process can be modified by the user.
    """

    def __init__(
        self,
        type: Literal["huber", "ransac"] = "huber",
        hyperparam_search_method: Literal["optuna", "grid"] | None = None,
        hyperparam_search_space: (
            Mapping[str, Iterable | BaseDistribution] | None
        ) = None,
        feature_selectors: list[BaseFSR] | None = None,
        max_n_features: int | None = None,
        model_random_state: int = 42,
        name: str | None = None,
        **kwargs,
    ):
        """
        Initializes a RobustLinearR object.

        Parameters
        ----------
        type : Literal['huber', 'ransac']
            Default: 'huber'.

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
            If None, the name is set to be the class name.

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

        super().__init__()
        self._dropfirst = True

        self._feature_selectors = feature_selectors
        self._max_n_features = max_n_features

        self._type = type

        if name is None:
            self._name = f"RobustLinearR({type})"
        else:
            self._name = name

        if type == "huber":
            self._best_estimator = HuberRegressor()
            if (hyperparam_search_method is None) or (hyperparam_search_space is None):
                hyperparam_search_method = "optuna"
                hyperparam_search_space = {
                    "epsilon": FloatDistribution(1.0, 2.0),
                    "alpha": FloatDistribution(1e-5, 1e1, log=True),
                }
        elif type == "ransac":
            self._best_estimator = RANSACRegressor(random_state=model_random_state)
            if (hyperparam_search_method is None) or (hyperparam_search_space is None):
                hyperparam_search_method = "optuna"
                hyperparam_search_space = {
                    "min_samples": FloatDistribution(0.1, 0.9),
                    "residual_threshold": FloatDistribution(1.0, 10.0),
                    "max_trials": CategoricalDistribution([100, 500, 1000]),
                }
        else:
            raise ValueError(f"Invalid value for type: {type}.")

        self._hyperparam_searcher = HyperparameterSearcher(
            estimator=self._best_estimator,
            method=hyperparam_search_method,
            hyperparam_grid=hyperparam_search_space,
            estimator_name=self._name,
            **kwargs,
        )

        self._validate_inputs()
