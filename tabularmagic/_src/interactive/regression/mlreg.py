import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable, Literal
from ...ml.discriminative.regression.base import BaseR
from ...data.datahandler import DataHandler
from ..visualization import plot_obs_vs_pred
from ...display.print_utils import print_wrapped


class SingleModelSingleDatasetMLRegReport:
    """
    Class for generating regression-relevant plots and
    tables for a single machine learning model on a single dataset.
    """

    def __init__(self, model: BaseR, dataset: Literal["train", "test"]):
        """
        Initializes a SingleModelSingleDatasetMLReport object.

        Parameters
        ----------
        model : BaseRegression.
            The data for the model must already be
            specified. The model should already be trained on the
            specified data.
        dataset : Literal['train', 'test'].
        """
        self.model = model
        if dataset not in ["train", "test"]:
            raise ValueError('dataset must be either "train" or "test".')
        self._dataset = dataset

    def fit_statistics(self) -> pd.DataFrame:
        """Returns a DataFrame containing the goodness-of-fit statistics
        for the model on the specified data.

        Returns
        ----------
        pd.DataFrame.
        """
        if self._dataset == "train":
            return self.model.train_scorer.stats_df()
        else:
            return self.model.test_scorer.stats_df()

    def cv_fit_statistics(self, averaged_across_folds: bool = True) -> pd.DataFrame:
        """Returns a DataFrame containing the cross-validated goodness-of-fit
        statistics for the model on the specified data.

        Parameters
        ----------
        averaged_across_folds.
            Default: True. If True, returns a DataFrame
            containing goodness-of-fit statistics averaged across all folds.
            Otherwise, returns a DataFrame containing goodness-of-fit
            statistics for each fold.

        Returns
        ----------
        pd.DataFrame.
        """
        if not self.model._is_cross_validated():
            print_wrapped(
                "Cross validation statistics are not available "
                + "for models that are not cross-validated.",
                type="WARNING",
            )
            return None
        if self._dataset == "train":
            if averaged_across_folds:
                return self.model.cv_scorer.stats_df()
            else:
                return self.model.cv_scorer.cv_stats_df()
        else:
            print_wrapped(
                "Cross validation statistics are not available for test data.",
                type="WARNING",
            )
            return None

    def plot_obs_vs_pred(
        self, figsize: Iterable = (5, 5), ax: plt.Axes | None = None
    ) -> plt.Figure:
        """Returns a figure that is a scatter plot of the observed (y-axis) and
        predicted (x-axis) values.

        Parameters
        ----------
        figsize : Iterable
            Default: (5, 5). The size of the figure.
        ax : plt.Axes.
            Default: None. The axes on which to plot the figure. If None,
            a new figure is created.

        Returns
        -------
        - plt.Figure
        """
        if self._dataset == "train":
            y_pred = self.model.train_scorer._y_pred
            y_true = self.model.train_scorer._y_true
        else:
            y_pred = self.model.test_scorer._y_pred
            y_true = self.model.test_scorer._y_true
        return plot_obs_vs_pred(y_pred, y_true, figsize, ax)


class SingleModelMLRegReport:
    """SingleModelMLRegReport: generates regression-relevant plots and
    tables for a single machine learning model.
    """

    def __init__(self, model: BaseR):
        """
        Initializes a SingleModelMLRegReport object.

        Parameters
        ----------
        - model : BaseRegression. The data for the model must already be
            specified. The model should already be trained on the
            specified data.
        """
        self.model = model

    def train_report(self) -> SingleModelSingleDatasetMLRegReport:
        """Returns a SingleModelSingleDatasetMLReport object for the training data.

        Returns
        -------
        - SingleModelSingleDatasetMLReport
        """
        return SingleModelSingleDatasetMLRegReport(self.model, "train")

    def test_report(self) -> SingleModelSingleDatasetMLRegReport:
        """Returns a SingleModelSingleDatasetMLReport object for the test data.

        Returns
        -------
        - SingleModelSingleDatasetMLReport
        """
        return SingleModelSingleDatasetMLRegReport(self.model, "test")


class MLRegressionReport:
    """Class for reporting model goodness of fit.
    Fits the model based on provided DataHandler.
    """

    def __init__(
        self,
        models: Iterable[BaseR],
        datahandler: DataHandler,
        y_var: str,
        X_vars: Iterable[str],
        outer_cv: int | None = None,
        outer_cv_seed: int = 42,
        verbose: bool = True,
    ):
        """MLRegressionReport.
        Fits the model based on provided DataHandler.

        Parameters
        ----------
        models : Iterable[BaseR].
            The BaseRegression models must already be trained.
        datahandler : DataHandler.
            The DataHandler object that contains the data.
        y_var : str.
            The name of the dependent variable.
        X_vars : Iterable[str].
            The names of the independent variables.
        outer_cv : int.
            Default: None.
            If not None, reports training scores via nested k-fold CV.
        outer_cv_seed : int.
            Default: 42. The random seed for the outer cross validation loop.
        verbose : bool.
            Default: True. If True, prints progress.
        """
        self._models: list[BaseR] = models
        self._id_to_model = {model._name: model for model in models}

        self.y_var = y_var
        self.X_vars = X_vars

        self._emitter = datahandler.train_test_emitter(y_var=y_var, X_vars=X_vars)
        self._emitters = None
        if outer_cv is not None:
            self._emitters = datahandler.kfold_emitters(
                y_var=y_var,
                X_vars=X_vars,
                n_folds=outer_cv,
                shuffle=True,
                random_state=outer_cv_seed,
            )

        self._verbose = verbose
        for model in self._models:
            if self._verbose:
                print_wrapped(f"Evaluating model {model._name}.", type="UPDATE")
            model.specify_data(dataemitter=self._emitter, dataemitters=self._emitters)
            model.fit(verbose=self._verbose)
            if self._verbose:
                print_wrapped(
                    f"Successfully evaluated model {model._name}.", type="UPDATE"
                )

        self._id_to_report = {
            model._name: SingleModelMLRegReport(model) for model in models
        }

    def model_report(self, model_id: str) -> SingleModelMLRegReport:
        """Returns the SingleModelMLRegReport object for the specified model.

        Parameters
        ----------
        model_id : str.
            The id of the model.

        Returns
        -------
        SingleModelMLRegReport
        """
        return self._id_to_report[model_id]

    def model(self, model_id: str) -> BaseR:
        """Returns the model with the specified id.

        Parameters
        ----------
        model_id : str.
            The id of the model.

        Returns
        -------
        BaseRegression
        """
        return self._id_to_model[model_id]

    def fit_statistics(
        self, dataset: Literal["train", "test"] = "test"
    ) -> pd.DataFrame:
        """Returns a DataFrame containing the goodness-of-fit statistics for
        all models on the specified data.

        Parameters
        ----------
        dataset : Literal['train', 'test'].
            Default: 'test'.

        Returns
        -------
        pd.DataFrame.
        """
        if dataset == "train":
            return pd.concat(
                [
                    report.train_report().fit_statistics()
                    for report in self._id_to_report.values()
                ],
                axis=1,
            )
        else:
            return pd.concat(
                [
                    report.test_report().fit_statistics()
                    for report in self._id_to_report.values()
                ],
                axis=1,
            )

    def cv_fit_statistics(self, averaged_across_folds: bool = True) -> pd.DataFrame:
        """Returns a DataFrame containing the cross-validated goodness-of-fit
        statistics for all models on the training data. Cross validation must
        have been conducted.

        Parameters
        ----------
        averaged_across_folds : bool.
            Default: True.
            If True, returns a DataFrame containing goodness-of-fit
            statistics averaged across all folds.
            Otherwise, returns a DataFrame containing goodness-of-fit
            statistics for each fold.

        Returns
        -------
        pd.DataFrame | None. None if cross validation was not conducted.
        """
        if not self._models[0]._is_cross_validated():
            print_wrapped(
                "Cross validation statistics are not available "
                + "for models that are not cross-validated.",
                type="WARNING",
            )
            return None
        return pd.concat(
            [
                report.train_report().cv_fit_statistics(averaged_across_folds)
                for report in self._id_to_report.values()
            ],
            axis=1,
        )

    def __getitem__(self, model_id: str) -> SingleModelMLRegReport:
        return self._id_to_report[model_id]
