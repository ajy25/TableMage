from typing import Literal
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np

from .base_feature_selection import BaseFSC, BaseFSR
from .BorutaPy import BorutaPy

from ..data.datahandler import DataEmitter
from ..display.print_utils import print_wrapped


class BorutaFSR(BaseFSR):
    def __init__(
        self,
        estimator: Literal["random_forest", "xgboost"] = "random_forest",
        n_estimators: int = 100,
        max_depth: int = 5,
        model_random_state: int = 42,
        n_jobs: int = -1,
        name: str | None = None,
    ):
        """
        Constructs a BorutaFSR.

        Parameters
        ----------
        estimator : Literal["random_forest", "xgboost"]
            Default: "random_forest". The estimator to use for Boruta. Default \
            hyperparameters are used for the estimator.

        n_estimators : int
            Default: 100. The number of estimators to use for Boruta.

        max_depth : int
            Default: 5. The maximum depth of the trees in the ensemble.

        model_random_state : int
            Default: 42. The random state to use for the estimator.

        n_jobs : int
            Default: -1. The number of jobs to run in parallel.

        name : str | None
            Default: None. If None, then outputs the default name.
        """
        if name is None:
            name = "BorutaFSR"
        super().__init__(name)

        sk_estimator = None
        if estimator == "random_forest":
            sk_estimator = RandomForestRegressor(
                max_depth=max_depth,
                random_state=model_random_state,
                n_jobs=n_jobs,
            )
        elif estimator == "xgboost":
            sk_estimator = XGBRegressor(
                max_depth=max_depth, random_state=model_random_state, n_jobs=n_jobs
            )
        else:
            raise ValueError(
                f"estimator must be one of 'random_forest', "
                f"or 'xgboost'. Got: {estimator}"
            )

        self._selector = BorutaPy(
            estimator=sk_estimator,
            n_estimators=n_estimators,
            random_state=model_random_state,
        )

    def select(
        self, dataemitter: DataEmitter
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Selects the top k features
        based on the training data.

        Parameters
        ----------
        dataemitter : DataEmitter

        Returns
        -------
        np.ndarray ~ (n_in_features)
            The selected features.

        np.ndarray ~ (n_in_features)
            The support of the selected features.

        np.ndarray ~ (n_in_features)
            The ranking of the selected features.
        """
        X_train, y_train = dataemitter.emit_train_Xy()
        self._all_features = X_train.columns.to_numpy()

        self._selector.fit(X=X_train.to_numpy(), y=y_train.to_numpy())
        self._support = self._selector.support_
        self._all_feature_scores = self._selector.ranking_
        self._selected_features = self._all_features[self._support]

        return self._all_features, self._selected_features, self._support


class BorutaFSC(BaseFSC):
    def __init__(
        self,
        estimator: Literal["random_forest", "xgboost"] = "random_forest",
        n_estimators: int = 100,
        max_depth: int = 5,
        model_random_state: int = 42,
        name: str | None = None,
    ):
        """
        Constructs a BorutaFSC.

        Parameters
        ----------
        estimator : Literal["random_forest", "xgboost"]
            Default: "random_forest". The estimator to use for Boruta. Default
            hyperparameters are used for the estimator.

        n_estimators : int
            Default: 100. The number of estimators to use for Boruta's estimator.

        max_depth : int
            Default: 5. The maximum depth of the trees in the ensemble.

        model_random_state : int
            Default: 42. The random state to use for the estimator.

        name : str | None
            Default: None. If None, then outputs the default name.
        """
        if name is None:
            name = "BorutaFSC"
        super().__init__(name)

        sk_estimator = None
        if estimator == "random_forest":
            sk_estimator = RandomForestClassifier(
                max_depth=max_depth,
                random_state=model_random_state,
                class_weight="balanced",
            )
        elif estimator == "xgboost":
            sk_estimator = XGBClassifier(
                max_depth=max_depth, random_state=model_random_state
            )
        else:
            raise ValueError(
                f"estimator must be one of 'random_forest', "
                f"or 'xgboost'. Got: {estimator}"
            )

        self._selector = BorutaPy(
            estimator=sk_estimator,
            n_estimators=n_estimators,
            random_state=model_random_state,
        )

    def select(
        self, dataemitter: DataEmitter
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Selects the top k features
        based on the training data.

        Parameters
        ----------
        dataemitter : DataEmitter

        Returns
        -------
        np.ndarray ~ (n_in_features)
            The selected features.

        np.ndarray ~ (n_in_features)
            The support of the selected features.

        np.ndarray ~ (n_in_features)
            The ranking of the selected features.
        """
        X_train, y_train = dataemitter.emit_train_Xy()
        y_train = LabelEncoder().fit_transform(y_train)
        self._all_features = X_train.columns.to_numpy()

        self._selector.fit(X=X_train.to_numpy(), y=y_train)
        self._support = self._selector.support_
        self._all_feature_scores = self._selector.ranking_
        self._selected_features = self._all_features[self._support]

        if len(self._selected_features) == 0:
            print_wrapped(
                "Boruta did not select any features. "
                "Boruta will vote for all features.",
                type="WARNING",
            )

        return self._all_features, self._selected_features, self._support
