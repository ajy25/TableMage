from sklearn.mixture import GaussianMixture
import pandas as pd
from typing import Literal
from ...display.print_utils import print_wrapped, quote_and_color
from .base_cluster import BaseClust


class GMMClust(BaseClust):
    """Class for Gaussian Mixture Model-based clustering."""

    def __init__(
        self,
        n_components: int | None = None,
        max_n_components: int = 10,
        criterion: Literal["bic", "aic"] = "bic",
        model_random_state: int = 42,
        name: str | None = None,
    ):
        """Initializes GMMClust.

        Parameters
        ----------
        n_components : int | None
            Number of components to fit.
            If None, the number of components is selected automatically
            based on the criterion.

        max_n_components : int
            Maximum number of components to fit.
            This parameter is only used when n_components is None.
            A grid search is performed from 1 to max_n_components to minimize
            the criterion.
            Default is 10.

        criterion : Literal["bic", "aic"]
            The criterion to use when selecting the number of components.
            The lower the better.
            Default is "bic".

        model_random_state : int
            Random state for the model. Default is 42.

        name : str | None
            Name of the model. Determines how
            the model is named in the report.
            If None, a default name is assigned.
        """
        super().__init__()
        self._n_components = n_components
        if name is not None:
            self._name = name
        else:
            if n_components is None:
                self._name = (
                    f"GMMClust(auto, max={max_n_components}, criterion={criterion})"
                )
            else:
                self._name = f"GMMClust({n_components})"
        self._max_n_components = max_n_components
        self._criterion = criterion
        self._model_random_state = model_random_state

    def fit(self, verbose: bool = False) -> None:
        """Fits the model to the data."""

        X_train = self._dataemitter.emit_train_X(dropfirst=False, verbose=verbose)
        X_test = self._dataemitter.emit_test_X(dropfirst=False, verbose=verbose)

        if isinstance(self._n_components, int):
            self._estimator = GaussianMixture(
                n_components=self._n_components,
                random_state=self._model_random_state,
            )

            self._n_clusters = self._n_components

            self._train_labels = self._estimator.fit_predict(X_train)
            self._test_labels = self._estimator.predict(X_test)

        else:
            best_n_components = 1
            best_metric = float("inf")  # the lower the better
            for n_components in range(1, self._max_n_components + 1):
                self._estimator = GaussianMixture(
                    n_components=n_components,
                    random_state=self._model_random_state,
                )
                self._estimator.fit(X_train)

                if self._criterion == "aic":
                    metric = self._estimator.aic(X_train)
                elif self._criterion == "bic":
                    metric = self._estimator.bic(X_train)
                if metric < best_metric:
                    best_metric = metric
                    best_n_components = n_components
            self._n_clusters = best_n_components
            self._estimator = GaussianMixture(
                n_components=best_n_components,
                random_state=self._model_random_state,
            )

            if verbose:
                print_wrapped(
                    f"Identified {self._n_clusters} optimal clusters for "
                    f"{quote_and_color(self._name, 'blue')} "
                    f"via {self._criterion}.",
                    type="PROGRESS",
                )

            self._train_labels = self._estimator.fit_predict(X_train)
            self._test_labels = self._estimator.predict(X_test)

        # convert labels to Series
        self._train_labels = pd.Series(
            self._train_labels, index=X_train.index, name="Label"
        )
        self._test_labels = pd.Series(
            self._test_labels, index=X_test.index, name="Label"
        )
