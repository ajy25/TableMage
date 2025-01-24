import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Any
from ...data.datahandler import DataHandler, DataEmitter
from ...metrics.visualization import decrease_font_sizes_axs
from ..logit import LogitLinearModel
from .linearreport_utils import MAX_N_OUTLIERS_TEXT, TRAIN_ONLY_MESSAGE
from ...display.print_options import print_options
from ...display.print_utils import (
    print_wrapped,
    color_text,
    bold_text,
    suppress_print_output,
    list_to_string,
    fill_ignore_format,
    format_two_column,
)
from ..lmutils.plot import (
    plot_residuals_vs_fitted,
    plot_scale_location,
    plot_residuals_vs_leverage,
    plot_qq,
    plot_pearson_vs_observation,
)

from ...metrics.visualization import plot_roc_curve, plot_confusion_matrix


class _SingleDatasetLogitReport:
    """Class for generating regression-relevant diagnostic
    plots and tables for a logistic regression model.
    """

    def __init__(self, model: LogitLinearModel, dataset: Literal["train", "test"]):
        """
        Initializes a _SingleDatasetLogitReport object.

        Parameters
        ----------
        model : LogitLinearModel
            The model must already be trained.

        dataset : Literal['train', 'test']
            The dataset to generate the report for.
        """
        self.model = model

        with suppress_print_output():
            if dataset == "test":
                self.scorer = model.test_scorer
                self._X_eval_df = self.model._dataemitter.emit_test_Xy()[0]
                self._is_train = False
            elif dataset == "train":
                self.scorer = model.train_scorer
                self._X_eval_df = self.model._dataemitter.emit_train_Xy()[0]
                self._is_train = True
            else:
                raise ValueError('specification must be either "train" or "test".')

        self._y_pred_score = self.scorer._y_pred_score

        # to obtain logit, inverse transform through the logistic function
        eps = 1e-12
        self._y_pred_score = np.clip(self._y_pred_score, eps, 1 - eps)
        self._y_pred_logit = np.log(self._y_pred_score / (1 - self._y_pred_score))

        self._y_pred = np.array(self.scorer._y_pred)
        self._y_true = np.array(self.scorer._y_true)

        # response residuals
        self._residuals = self._y_true - self._y_pred_score

        # Pearson residuals
        self._stdresiduals = self._residuals / np.sqrt(
            self._y_pred_score * (1 - self._y_pred_score) + eps
        )

        self._outlier_threshold = 2
        self._compute_outliers()

        self._include_text = False
        if self._n_outliers <= MAX_N_OUTLIERS_TEXT:
            self._include_text = True

    def set_outlier_threshold(self, threshold: float) -> "_SingleDatasetLogitReport":
        """Standardized residuals threshold for outlier identification.
        Recomputes the outliers.

        Parameters
        ----------
        threshold : float
            Default: 2. Must be a nonnegative value.

        Returns
        -------
        self
        """
        if threshold < 0:
            raise ValueError(
                f"Input threshold must be nonnegative. Received {threshold}."
            )
        self._outlier_threshold = threshold
        self._compute_outliers()
        return self

    def get_outlier_indices(self) -> list:
        """Returns the indices corresponding to DataFrame examples associated
        with standardized residual outliers.

        Returns
        -------
        outliers_df_idx : list ~ (n_outliers)
        """
        return self._outliers_df_idx.tolist()

    def metrics(self) -> pd.DataFrame:
        """Returns a DataFrame containing the goodness-of-fit statistics
        for the model.

        Parameters
        ----------
        pd.DataFrame
        """
        return self.scorer.stats_df().astype(float)

    def plot_confusion_matrix(
        self, figsize: tuple[float, float] = (5, 5), ax: plt.Axes | None = None
    ) -> plt.Figure:
        """Returns a figure that is the confusion matrix for the model.

        Parameters
        ----------
        figsize: tuple[float, float]
            Default: (5, 5). The size of the figure.

        ax: plt.Axes
            Default: None. The axes on which to plot the figure. If None,
            a new figure is created.

        Returns
        -------
        plt.Figure
            Figure of the confusion matrix.
        """
        y_pred = self.scorer._y_pred
        y_true = self.scorer._y_true
        return plot_confusion_matrix(
            y_pred=y_pred,
            y_true=y_true,
            model_name=self.model._name,
            figsize=figsize,
            ax=ax,
        )

    def plot_roc_curve(
        self,
        label_curve: bool = False,
        color: str | Any = None,
        figsize: tuple[float, float] = (5, 5),
        ax: plt.Axes | None = None,
    ) -> plt.Figure | None:
        """Returns a figure that is the ROC curve for the model.

        Parameters
        ----------
        label_curve : bool
            Default: False. Whether to label the ROC curve with model name and AUC.
            If True, the model name and AUC are displayed on the ROC curve rather
            than in the title. This is useful when plotting multiple ROC curves
            on the same axes.

        color : str | Any
            Default: None. The color of the ROC curve. The color of the ROC curve.
            If None, the plot options line color is used.

        figsize: tuple[float, float]
            Default: (5, 5). The size of the figure.

        ax: plt.Axes | None
            Default: None. The axes on which to plot the figure. If None,
            a new figure is created.

        Returns
        -------
        plt.Figure | None
            Figure of the ROC curve. None is returned if the model is not binary.
        """
        y_score = self.scorer._y_pred_score
        y_true = self.scorer._y_true
        return plot_roc_curve(
            y_score=y_score,
            y_true=y_true,
            model_name=self.model._name,
            label_curve=label_curve,
            color=color,
            figsize=figsize,
            ax=ax,
        )

    def plot_scale_location(
        self,
        show_outliers: bool = True,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a figure that is a plot of the
        sqrt of the residuals versus the fitted.

        Parameters
        ----------
        show_outliers : bool
            Default: True. If True, plots the outliers in red.

        figsize : tuple[float, float]
            Default: (5.0, 5.0).

        ax : plt.Axes
            Default: None.

        Returns
        -------
        plt.Figure
        """
        fig = plot_scale_location(
            y_pred=self._y_pred_logit,
            std_residuals=self._stdresiduals,
            show_outliers=show_outliers,
            outliers_idx=self._outliers_df_idx,
            outliers_mask=self._outliers_residual_mask,
            include_text=self._include_text,
            figsize=figsize,
            ax=ax,
        )
        if ax is None:
            ax = fig.gca()
        ax.set_ylabel("Pearson Residuals")
        return fig

    def plot_residuals_vs_leverage(
        self,
        standardized: bool = True,
        show_outliers: bool = True,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a figure that is a plot of the residuals versus leverage.

        Parameters
        ----------
        standardized : bool
            Default: True. If True, standardizes the residuals.

        show_outliers : bool
            Default: True. If True, plots the outliers in red.

        figsize : tuple[float, float]
            Default: (5.0, 5.0).

        ax : plt.Axes
            Default: None.

        Returns
        -------
        plt.Figure
        """
        if not self._is_train:
            print_wrapped(TRAIN_ONLY_MESSAGE, type="WARNING")
            return None
        leverage = self.model.estimator._results.get_influence().hat_matrix_diag

        return plot_residuals_vs_leverage(
            leverage=leverage,
            residuals=self._residuals,
            standardized=standardized,
            show_outliers=show_outliers,
            outliers_idx=self._outliers_df_idx,
            outliers_mask=self._outliers_residual_mask,
            include_text=self._include_text,
            figsize=figsize,
            ax=ax,
        )

    def plot_qq(
        self,
        standardized: bool = True,
        show_outliers: bool = False,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a quantile-quantile plot.

        Parameters
        ----------
        standardized : bool
            Default: True. If True, standardizes the residuals.

        show_outliers : bool
            Default: False. If True, plots the outliers in red.

        figsize : tuple[float, float]
            Default: (5.0, 5.0).

        ax : plt.Axes
            Default: None.

        Returns
        -------
        plt.Figure
        """
        return plot_qq(
            df_idx=self._X_eval_df.index,
            residuals=self._residuals,
            standardized=standardized,
            outliers_idx=self._outliers_df_idx,
            outliers_mask=self._outliers_residual_mask,
            show_outliers=show_outliers,
            include_text=self._include_text,
            figsize=figsize,
            ax=ax,
        )

    def plot_residuals_vs_fitted(
        self,
        standardized: bool = False,
        show_outliers: bool = True,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a figure that is a residuals vs fitted (y_pred) plot.

        Parameters
        ----------
        standardized : bool
            Default: False. If True, plots the standardized residuals as
            opposed to the raw residuals.

        show_outliers : bool
            Default: True. If True, colors the outliers determined by the
            standardized residuals in red.

        figsize : tuple[float, float]
            Default: (5.0, 5.0). Determines the size of the returned figure.

        ax : plt.Axes
            Default: None.

        Returns
        -------
        plt.Figure
        """
        return plot_residuals_vs_fitted(
            y_pred=self._y_pred_logit,
            residuals=self._residuals,
            outliers_idx=self._outliers_df_idx,
            outliers_mask=self._outliers_residual_mask,
            show_outliers=show_outliers,
            standardized=standardized,
            include_text=self._include_text,
            figsize=figsize,
            ax=ax,
        )

    def plot_pearson_vs_observation(
        self,
        show_outliers: bool = True,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Plots Pearson residuals against observation number.

        Parameters
        ----------
        show_outliers : bool
            Default: True. If True, highlights the outliers in red.

        figsize : tuple[float, float]
            Default: (5.0, 5.0). Determines the size of the returned figure.

        ax : plt.Axes
            Default: None. The axes on which to plot the figure. If None,
            a new figure is created.

        Returns
        -------
        plt.Figure
            The resulting plot as a figure.
        """
        fig = plot_pearson_vs_observation(
            std_residuals=self._stdresiduals,
            df_idx=self._X_eval_df.index,
            outliers_idx=self._outliers_df_idx,
            outliers_mask=self._outliers_residual_mask,
            show_outliers=show_outliers,
            include_text=self._include_text,
            figsize=figsize,
            ax=ax,
        )
        if ax is None:
            ax = fig.gca()
        ax.set_ylabel("Pearson Residuals")
        return fig

    def plot_diagnostics(
        self, show_outliers: bool = False, figsize: tuple[float, float] = (7.0, 7.0)
    ) -> plt.Figure:
        """Plots several useful linear regression diagnostic plots.

        Parameters
        ----------
        show_outliers : bool
            Default: False. If True, plots the residual outliers in red.

        figsize : tuple[float, float]
            Default: (7.0, 7.0).

        Returns
        -------
        plt.Figure
        """
        fig, axs = plt.subplots(2, 2, figsize=figsize)

        self.plot_residuals_vs_fitted(show_outliers=show_outliers, ax=axs[0][0])
        self.plot_scale_location(show_outliers=show_outliers, ax=axs[1][0])
        if self._is_train:
            self.plot_residuals_vs_leverage(show_outliers=show_outliers, ax=axs[1][1])
        else:
            self.plot_roc_curve(ax=axs[1][1])

        self.plot_qq(show_outliers=show_outliers, ax=axs[0][1])

        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        decrease_font_sizes_axs(axs, 2, 2, 0)

        plt.close(fig)
        return fig

    def _compute_outliers(self):
        """Computes the outliers."""
        self._outliers_residual_mask = (
            self._stdresiduals >= self._outlier_threshold
        ) | (self._stdresiduals <= -self._outlier_threshold)
        self._outliers_df_idx = self._X_eval_df.iloc[
            self._outliers_residual_mask
        ].index.to_numpy()
        self._n_outliers = len(self._outliers_df_idx)
        self._include_text = False
        if self._n_outliers <= MAX_N_OUTLIERS_TEXT:
            self._include_text = True


class LogitReport:
    """LogitReport.
    Fits the model based on provided DataHandler.
    Wraps train and test SingleDatasetLogitReport objects.
    """

    def __init__(
        self,
        model: LogitLinearModel,
        datahandler: DataHandler,
        target: str,
        predictors: list[str],
        dataemitter: DataEmitter | None = None,
    ):
        """LogitReport.
        Fits the model based on provided DataHandler.
        Wraps train and test SingleDatasetLogitReport objects.

        Parameters
        ----------
        model : LogitLinearModel

        datahandler : DataHandler
            The DataHandler object that contains the data.

        target : str
            The name of the dependent variable.

        predictors : list[str]
            The names of the independent variables.
        """
        self._model = model
        self._datahandler = datahandler
        if dataemitter is not None:
            self._dataemitter = dataemitter
        else:
            self._dataemitter = self._datahandler.train_test_emitter(target, predictors)
        self._model.specify_data(self._dataemitter)
        self._model.fit()
        self._target = target
        self._predictors = predictors
        self._train_report = _SingleDatasetLogitReport(model, "train")
        self._test_report = _SingleDatasetLogitReport(model, "test")

    def train_report(self) -> _SingleDatasetLogitReport:
        """Returns a SingleDatasetLogitReport object for the train dataset

        Returns
        -------
        SingleDatasetLogitReport
        """
        return self._train_report

    def test_report(self) -> _SingleDatasetLogitReport:
        """Returns a SingleDatasetLogitReport object for the test dataset

        Returns
        -------
        SingleDatasetLogitReport
        """
        return self._test_report

    def model(self) -> LogitLinearModel:
        """Returns the fitted LogitLinearModel object.

        Returns
        -------
        LogitLinearModel
        """
        return self._model

    def metrics(self, dataset: Literal["train", "test", "both"]) -> pd.DataFrame:
        """Returns a DataFrame containing the goodness-of-fit statistics
        for the model.

        Parameters
        ----------
        dataset : Literal['train', 'test', 'both']
            The dataset to get the metrics for.

        Returns
        -------
        pd.DataFrame
        """
        if dataset == "train":
            return self._train_report.metrics()
        elif dataset == "test":
            return self._test_report.metrics()
        elif dataset == "both":
            test_metrics = self._test_report.metrics()
            train_metrics = self._train_report.metrics()
            return pd.concat(
                [train_metrics, test_metrics], keys=["train", "test"], names=["Dataset"]
            )
        else:
            raise ValueError('dataset must be either "train", "test", or "both".')

    def step(
        self,
        direction: Literal["both", "backward", "forward"] = "backward",
        criteria: Literal["aic", "bic"] = "aic",
        kept_vars: list[str] | None = None,
        all_vars: list[str] | None = None,
        start_vars: list[str] | None = None,
        max_steps: int = 100,
    ) -> "LogitReport":
        """Performs stepwise selection on the model. Returns a new
        LogitReport object with the updated model.

        Parameters
        ----------
        direction : Literal["both", "backward", "forward"]
            Default: 'backward'. The direction of the stepwise selection.

        criteria : Literal["aic", "bic"]
            Default: 'aic'. The criteria to use for selecting the best model.

        kept_vars : list[str]
            Default: None. The variables that should be kept in the model.
            If None, defaults to empty list.

        all_vars : list[str]
            Default: None. The variables that are candidates for inclusion in the model.
            If None, defaults to all variables in the training data.

        start_vars : list[str]
            Default: None.
            The variables to start the bidirectional stepwise selection with.
            Ignored if direction is not 'both'. If direction is 'both' and
            start_vars is None, then the starting variables are the kept_vars.

        max_steps : int
            Default: 100. The maximum number of steps to take.

        Returns
        -------
        LogitReport
        """
        if direction == "backward":
            method_name = "Backward selection"
        elif direction == "both":
            method_name = "Alternating selection"
        elif direction == "forward":
            method_name = "Forward selection"
        else:
            raise ValueError(f"Invalid argument: {direction}.")
        selected_vars = self._model.step(
            direction=direction,
            criteria=criteria,
            kept_vars=kept_vars,
            all_vars=all_vars,
            start_vars=start_vars,
            max_steps=max_steps,
        )

        if all_vars is None:
            all_vars = self._model._dataemitter.X_vars()
        vars_removed = list(set(all_vars) - set(selected_vars))
        if len(vars_removed) == 0:
            print_wrapped(
                f"{method_name} removed 0 predictors.", level="INFO", type="NOTE"
            )
            return self
        elif len(vars_removed) == 1:
            print_wrapped(
                text=f"{method_name} removed {len(vars_removed)} predictor: "
                + list_to_string(vars_removed)
                + ".",
                level="INFO",
                type="UPDATE",
            )
        else:
            print_wrapped(
                text=f"{method_name} removed {len(vars_removed)} predictors: "
                + list_to_string(vars_removed)
                + ".",
                level="INFO",
                type="UPDATE",
            )

        new_emitter = self._dataemitter.copy()
        new_emitter.select_predictors_pre_onehot(selected_vars)

        return LogitReport(
            LogitLinearModel(
                alpha=self._model.alpha,
                l1_weight=self._model.l1_weight,
                threshold_strategy=self._model._threshold_strategy,
                name=self._model._name + " (Reduced)",
            ),
            self._datahandler,
            self._target,
            selected_vars,
            new_emitter,
        )

    def statsmodels_summary(self):
        """Returns the summary of the statsmodels RegressionResultsWrapper for
        OLS.
        """
        try:
            return self._model.estimator.summary()
        except Exception as e:
            raise RuntimeError(
                "Error occured in statsmodels_summary call. " f"Error: {e}"
            )

    def set_outlier_threshold(
        self, threshold: float, dataset: Literal["train", "test"] = "test"
    ) -> "_SingleDatasetLogitReport":
        """Standardized residuals threshold for outlier identification.
        Recomputes the outliers.

        Parameters
        ----------
        threshold : float
            Default: 2. Must be a nonnegative value.

        dataset : Literal['train', 'test']
            Default: 'test'.

        Returns
        -------
        self
        """
        if dataset == "train":
            self._train_report.set_outlier_threshold(threshold=threshold)
        else:
            self._test_report.set_outlier_threshold(threshold=threshold)

    def get_outlier_indices(self, dataset: Literal["train", "test"] = "test") -> list:
        """Returns the indices corresponding to DataFrame examples associated
        with standardized residual outliers.

        Parameters
        ----------
        dataset : Literal['train', 'test']
            Default: 'test'.

        Returns
        -------
        outliers_df_idx : list ~ (n_outliers)
        """
        if dataset == "train":
            return self._train_report.get_outlier_indices()
        else:
            return self._test_report.get_outlier_indices()

    def coefs(
        self,
        format: Literal[
            "coef(se)|pval", "coef|se|pval", "coef(ci)|pval", "coef|ci_low|ci_high|pval"
        ] = "coef(se)|pval",
    ) -> pd.DataFrame:
        """Returns a DataFrame containing the coefficients of the model.

        Parameters
        ----------
        format : Literal["coef(se)|pval", "coef|se|pval", "coef(ci)|pval",
                        "coef|ci_low|ci_high|pval"]
            Default: 'coef(se)|pval'.

        Returns
        -------
        pd.DataFrame
        """
        return self._model.coefs(format=format)

    def plot_residuals_vs_fitted(
        self,
        dataset: Literal["train", "test"],
        standardized: bool = False,
        show_outliers: bool = True,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Plots the residuals versus the fitted values.

        Parameters
        ----------
        dataset : Literal['train', 'test']
            The dataset to generate the plot for.

        standardized : bool
            Default: False. If True, plots the standardized residuals as
            opposed to the raw residuals.

        show_outliers : bool
            Default: True. If True, colors the outliers determined by the
            standardized residuals in red.

        figsize : tuple[float, float]
            Default: (5.0, 5.0). Determines the size of the returned figure.

        ax : plt.Axes
            Default: None.

        Returns
        -------
        plt.Figure
        """
        if dataset == "train":
            return self._train_report.plot_residuals_vs_fitted(
                standardized=standardized,
                show_outliers=show_outliers,
                figsize=figsize,
                ax=ax,
            )
        elif dataset == "test":
            return self._test_report.plot_residuals_vs_fitted(
                standardized=standardized,
                show_outliers=show_outliers,
                figsize=figsize,
                ax=ax,
            )
        else:
            raise ValueError('The dataset must be either "train" or "test".')

    def plot_pearson_vs_observation(
        self,
        dataset: Literal["train", "test"],
        show_outliers: bool = True,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Plots Pearson residuals against observation number.

        Parameters
        ----------
        show_outliers : bool
            Default: True. If True, highlights the outliers in red.

        figsize : tuple[float, float]
            Default: (5.0, 5.0). Determines the size of the returned figure.

        ax : plt.Axes
            Default: None. The axes on which to plot the figure. If None,
            a new figure is created.

        Returns
        -------
        plt.Figure
            The resulting plot as a figure.
        """

        if dataset == "train":
            return self._train_report.plot_pearson_vs_observation(
                show_outliers=show_outliers,
                figsize=figsize,
                ax=ax,
            )
        elif dataset == "test":
            return self._test_report.plot_pearson_vs_observation(
                show_outliers=show_outliers,
                figsize=figsize,
                ax=ax,
            )
        else:
            raise ValueError('The dataset must be either "train" or "test".')

    def plot_scale_location(
        self,
        dataset: Literal["train", "test"],
        show_outliers: bool = True,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a figure that is a plot of the
        sqrt of the residuals versus the fitted.

        Parameters
        ----------
        dataset : Literal['train', 'test']
            The dataset to generate the plot for.

        show_outliers : bool
            Default: True. If True, plots the outliers in red.

        figsize : tuple[float, float]
            Default: (5.0, 5.0).

        ax : plt.Axes
            Default: None.

        Returns
        -------
        plt.Figure
        """
        if dataset == "train":
            return self._train_report.plot_scale_location(
                show_outliers=show_outliers, figsize=figsize, ax=ax
            )
        elif dataset == "test":
            return self._test_report.plot_scale_location(
                show_outliers=show_outliers, figsize=figsize, ax=ax
            )
        else:
            raise ValueError('The dataset must be either "train" or "test".')

    def plot_residuals_vs_leverage(
        self,
        dataset: Literal["train", "test"],
        standardized: bool = True,
        show_outliers: bool = True,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Plots the residuals versus leverage.

        Parameters
        ----------
        dataset : Literal['train', 'test']
            Default: 'test'.

        standardized : bool
            Default: True. If True, standardizes the residuals.

        show_outliers : bool
            Default: True. If True, plots the outliers in red.

        figsize : tuple[float, float]
            Default: (5.0, 5.0).

        ax : plt.Axes
            Default: None.

        Returns
        -------
        plt.Figure
        """
        if dataset == "train":
            return self._train_report.plot_residuals_vs_leverage(
                standardized=standardized,
                show_outliers=show_outliers,
                figsize=figsize,
                ax=ax,
            )
        elif dataset == "test":
            return self._test_report.plot_residuals_vs_leverage(
                standardized=standardized,
                show_outliers=show_outliers,
                figsize=figsize,
                ax=ax,
            )
        else:
            raise ValueError('The dataset must be either "train" or "test".')

    def plot_qq(
        self,
        dataset: Literal["train", "test"],
        standardized: bool = True,
        show_outliers: bool = False,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Plots a quantile-quantile plot of the residuals.

        Parameters
        ----------
        dataset : Literal['train', 'test']
            The dataset to generate the plot for.

        standardized : bool
            Default: True. If True, standardizes the residuals.

        show_outliers : bool
            Default: False. If True, plots the outliers in red.

        figsize : tuple[float, float]
            Default: (5.0, 5.0).

        ax : plt.Axes
            Default: None.

        Returns
        -------
        plt.Figure
        """
        if dataset == "train":
            return self._train_report.plot_qq(
                standardized=standardized,
                show_outliers=show_outliers,
                figsize=figsize,
                ax=ax,
            )
        elif dataset == "test":
            return self._test_report.plot_qq(
                standardized=standardized,
                show_outliers=show_outliers,
                figsize=figsize,
                ax=ax,
            )
        else:
            raise ValueError('The dataset must be either "train" or "test".')

    def plot_diagnostics(
        self,
        dataset: Literal["train", "test"],
        show_outliers: bool = False,
        figsize: tuple[float, float] = (7.0, 7.0),
    ) -> plt.Figure:
        """Plots several useful linear regression diagnostic plots.

        Parameters
        ----------
        dataset : Literal['train', 'test']
            The dataset to generate the plot for.

        show_outliers : bool
            Default: False. If True, plots the residual outliers in red.

        figsize : tuple[float, float]
            Default: (7.0, 7.0).

        Returns
        -------
        plt.Figure
        """
        if dataset == "train":
            return self._train_report.plot_diagnostics(
                show_outliers=show_outliers, figsize=figsize
            )
        elif dataset == "test":
            return self._test_report.plot_diagnostics(
                show_outliers=show_outliers, figsize=figsize
            )
        else:
            raise ValueError('The dataset must be either "train" or "test".')

    def plot_roc_curve(
        self,
        dataset: Literal["train", "test"],
        label_curve: bool = False,
        color: str | Any = None,
        figsize: tuple[float, float] = (5, 5),
        ax: plt.Axes | None = None,
    ) -> plt.Figure | None:
        """Returns a figure that is the ROC curve for the model.

        Parameters
        ----------
        dataset : Literal['train', 'test']
            The dataset to generate the plot for.

        label_curve : bool
            Default: False. Whether to label the ROC curve with model name and AUC.

        color : str | Any
            Default: None. The color of the ROC curve. The color of the ROC curve.

        figsize: tuple[float, float]
            Default: (5, 5). The size of the figure.

        ax: plt.Axes | None
            Default: None. The axes on which to plot the figure. If None,
            a new figure is created.

        Returns
        -------
        plt.Figure | None
            Figure of the ROC curve.
        """
        if dataset == "train":
            return self._train_report.plot_roc_curve(
                label_curve=label_curve,
                color=color,
                figsize=figsize,
                ax=ax,
            )
        elif dataset == "test":
            return self._test_report.plot_roc_curve(
                label_curve=label_curve,
                color=color,
                figsize=figsize,
                ax=ax,
            )
        else:
            raise ValueError('The dataset must be either "train" or "test".')

    def plot_confusion_matrix(
        self,
        dataset: Literal["train", "test"],
        figsize: tuple[float, float] = (5, 5),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a figure that is the confusion matrix for the model.

        Parameters
        ----------
        dataset : Literal['train', 'test']
            The dataset to generate the plot for.

        figsize: tuple[float, float]
            Default: (5, 5). The size of the figure.

        ax: plt.Axes
            Default: None. The axes on which to plot the figure. If None,

        Returns
        -------
        plt.Figure
        """

        if dataset == "train":
            return self._train_report.plot_confusion_matrix(
                figsize=figsize,
                ax=ax,
            )
        elif dataset == "test":
            return self._test_report.plot_confusion_matrix(
                figsize=figsize,
                ax=ax,
            )
        else:
            raise ValueError('The dataset must be either "train" or "test".')

    def _compute_outliers(self, dataset: Literal["train", "test"] = "test"):
        """Computes the outliers.

        Parameters
        ----------
        dataset : Literal['train', 'test']
            Default: 'test'.
        """
        if dataset == "train":
            return self._train_report._compute_outliers()
        else:
            return self._test_report._compute_outliers()

    def _to_dict(self) -> dict:
        """Returns a dictionary representation of the LogitReport object.

        Returns
        -------
        dict
        """
        output = {
            "train_metrics": self.metrics("train").to_dict("index"),
            "test_metrics": self.metrics("test").to_dict("index"),
        }
        if hasattr(self, "coefs"):
            if callable(self.coefs):
                output["coefficients"] = self.coefs().to_dict("index")
        return output

    def __str__(self) -> str:
        max_width = print_options._max_line_width
        n_dec = print_options._n_decimals

        top_divider = color_text("=" * max_width, "none") + "\n"
        bottom_divider = "\n" + color_text("=" * max_width, "none")
        divider = "\n" + color_text("-" * max_width, "none") + "\n"
        divider_invisible = "\n" + " " * max_width + "\n"

        if self._model.alpha == 0:
            title_message = bold_text("Logistic Regression Report")
        else:
            if self._model.l1_weight == 0:
                title_message = bold_text(
                    f"Logistic Regression Report (l2_penalty={self._model.alpha})"
                )
            elif self._model.l1_weight == 1:
                title_message = bold_text(
                    f"Logistic Regression Report (l1_penalty={self._model.alpha})"
                )
            else:
                title_message = bold_text(
                    f"Elastic Net Report (alpha={self._model.alpha}, "
                    f"l1_ratio={self._model.l1_weight})"
                )

        target_var = "'" + self._target + "'"
        target_message = f"{bold_text('Target variable:')}\n"
        target_message += fill_ignore_format(
            color_text(target_var, "purple"),
            width=max_width,
            initial_indent=2,
            subsequent_indent=2,
        )

        predictors_message = (
            f"{bold_text(f'Predictor variables ({len(self._predictors)}):')}\n"
        )
        predictors_message += fill_ignore_format(
            list_to_string(self._predictors),
            width=max_width,
            initial_indent=2,
            subsequent_indent=2,
        )

        metrics_message = f"{bold_text('Metrics:')}\n"
        metrics_message += fill_ignore_format(
            format_two_column(
                bold_text(f"Train ({self._model._n_train})"),
                bold_text(f"Test ({self._model._n_test})"),
                total_len=max_width - 2,
            ),
            initial_indent=2,
        )
        mstr = str(self._model)
        metrics_message += "\n"
        metrics_message += fill_ignore_format(
            format_two_column(
                "F1:     "
                + color_text(
                    str(np.round(self.metrics("train").at["f1", mstr], n_dec)), "yellow"
                ),
                "F1:     "
                + color_text(
                    str(np.round(self.metrics("test").at["f1", mstr], n_dec)), "yellow"
                ),
                total_len=max_width - 2,
            ),
            initial_indent=4,
        )
        metrics_message += "\n"
        metrics_message += fill_ignore_format(
            format_two_column(
                "Acc:    "
                + color_text(
                    str(np.round(self.metrics("train").at["accuracy", mstr], n_dec)),
                    "yellow",
                ),
                "Acc:    "
                + color_text(
                    str(np.round(self.metrics("test").at["accuracy", mstr], n_dec)),
                    "yellow",
                ),
                total_len=max_width - 2,
            ),
            initial_indent=4,
        )
        metrics_message += "\n"
        metrics_message += fill_ignore_format(
            format_two_column(
                "AUROC:  "
                + color_text(
                    str(np.round(self.metrics("train").at["roc_auc", mstr], n_dec)),
                    "yellow",
                ),
                "AUROC:  "
                + color_text(
                    str(np.round(self.metrics("test").at["roc_auc", mstr], n_dec)),
                    "yellow",
                ),
                total_len=max_width - 2,
            ),
            initial_indent=4,
        )

        col_ratios = [3, 4, 3, 3]
        col_space = [max_width // sum(col_ratios) * i for i in col_ratios]

        coefs_df = self.coefs("coef|se|pval")
        coefs_df.index.name = "Predictor"

        coefs_message = bold_text("Coefficients:") + "\n"
        actual_coefs_message = fill_ignore_format(
            coefs_df.to_string(col_space=col_space[1:]),
            width=max_width,
            initial_indent=2,
            subsequent_indent=2,
        )
        # bold the first two lines of the actual_coefs_message
        actual_coefs_message = "\n".join(
            [
                bold_text(line) if i in [0, 1] else line
                for i, line in enumerate(actual_coefs_message.split("\n"))
            ]
        )
        coefs_message += actual_coefs_message

        final_message = (
            top_divider
            + title_message
            + divider
            + target_message
            + divider_invisible
            + predictors_message
            + divider
            + metrics_message
            + divider
            + coefs_message
            + bottom_divider
        )

        return final_message

    def _repr_pretty_(self, p, cycle) -> str:
        p.text(str(self))
