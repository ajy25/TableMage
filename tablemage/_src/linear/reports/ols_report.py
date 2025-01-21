import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
import warnings
from adjustText import adjust_text
from statsmodels.regression.linear_model import OLSResults, RegressionResultsWrapper
from ...data import DataHandler, DataEmitter
from ...metrics.visualization import plot_obs_vs_pred, decrease_font_sizes_axs
from ..ols import OLSLinearModel
from ...display.print_utils import print_wrapped, suppress_print_output
from .linearreport_utils import MAX_N_OUTLIERS_TEXT, TRAIN_ONLY_MESSAGE
from ..lmutils.plot import (
    plot_residuals_vs_var,
    plot_residuals_vs_fitted,
    plot_residuals_hist,
    plot_scale_location,
    plot_residuals_vs_leverage,
    plot_qq,
)
from ...display.print_options import print_options
from ...display.plot_options import plot_options
from ...display.print_utils import (
    print_wrapped,
    color_text,
    bold_text,
    list_to_string,
    fill_ignore_format,
    format_two_column,
)
from ...stattests import StatisticalTestReport


class _SingleDatasetOLSReport:
    """Class for generating regression-relevant diagnostic
    plots and tables for a single linear regression model.
    """

    def __init__(self, model: OLSLinearModel, dataset: Literal["train", "test"]):
        """
        Initializes a _SingleDatasetOLSReport object.

        Parameters
        ----------
        model : OLSLinearModel.
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

        self._y_pred = self.scorer._y_pred
        self._y_true = self.scorer._y_true

        self._residuals = self._y_true - self._y_pred
        self._stdresiduals = self._residuals / np.std(self._residuals)
        self._outlier_threshold = 2
        self._compute_outliers()

        self._include_text = False
        if self._n_outliers <= MAX_N_OUTLIERS_TEXT:
            self._include_text = True

    def plot_obs_vs_pred(
        self,
        show_outliers: bool = True,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a figure that is a scatter plot of the true and predicted y
        values.

        Parameters
        ----------
        show_outliers : bool
            Default: True.
            If True, then the outliers calculated using standard errors will be
            shown in red.

        figsize : tuple[float, float]
            Default: (5.0,5.0). Sets the size of the resulting graph.

        ax : plt.Axes
            Default: None.

        Returns
        -------
        plt.Figure
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        plot_obs_vs_pred(self._y_pred, self._y_true, self.model._name, figsize, ax)
        if show_outliers and self._n_outliers > 0:
            ax.scatter(
                self._y_pred[self._outliers_residual_mask],
                self._y_true[self._outliers_residual_mask],
                s=plot_options._dot_size,
                color="red",
            )
            if self._include_text and self._n_outliers <= MAX_N_OUTLIERS_TEXT:
                annotations = []
                for i, label in enumerate(self._outliers_df_idx):
                    annotations.append(
                        ax.annotate(
                            label,
                            (
                                self._y_pred[self._outliers_residual_mask][i],
                                self._y_true[self._outliers_residual_mask][i],
                            ),
                            color="red",
                            fontsize=plot_options._axis_minor_ticklabel_font_size,
                        )
                    )
                adjust_text(annotations, ax=ax)

        if fig is not None:
            fig.tight_layout()
            plt.close()
        return fig

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
            y_pred=self._y_pred,
            residuals=self._residuals,
            outliers_idx=self._outliers_df_idx,
            outliers_mask=self._outliers_residual_mask,
            show_outliers=show_outliers,
            standardized=standardized,
            include_text=self._include_text,
            figsize=figsize,
            ax=ax,
        )

    def plot_residuals_vs_var(
        self,
        predictor: str,
        standardized: bool = False,
        show_outliers: bool = False,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a figure that is a residuals vs fitted (y_pred) plot.

        Parameters
        ----------
        predictor : str
            The predictor variable whose values should be plotted on the x-axis.

        standardized : bool
            Default: False. If True, standardizes the residuals.

        show_outliers : bool
            Default: False. If True, plots the outliers in red.

        figsize : tuple[float, float]
            Default: (5.0, 5.0). Determines the size of the returned figure.

        ax : plt.Axes
            Default: None.

        Returns
        -------
        plt.Figure
        """
        return plot_residuals_vs_var(
            predictor=predictor,
            X_eval_df=self._X_eval_df,
            residuals=self._residuals,
            outliers_idx=self._outliers_df_idx,
            outliers_mask=self._outliers_residual_mask,
            show_outliers=show_outliers,
            standardized=standardized,
            include_text=self._include_text,
            figsize=figsize,
            ax=ax,
        )

    def plot_residuals_hist(
        self,
        standardized: bool = False,
        density: bool = False,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a figure that is a histogram of the residuals.

        Parameters
        ----------
        standardized : bool
            Default: False. If True, standardizes the residuals.

        density : bool
            Default: False. If True, plots density rather than frequency.

        figsize : tuple[float, float]
            Default: (5.0, 5.0). Determines the size of the returned figure.

        ax : plt.Axes
            Default: None.

        Returns
        -------
        plt.Figure
        """
        return plot_residuals_hist(
            residuals=self._residuals,
            standardized=standardized,
            density=density,
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
        return plot_scale_location(
            y_pred=self._y_pred,
            std_residuals=self._residuals / np.std(self._residuals),
            show_outliers=show_outliers,
            outliers_idx=self._outliers_df_idx,
            outliers_mask=self._outliers_residual_mask,
            include_text=self._include_text,
            figsize=figsize,
            ax=ax,
        )

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

        if isinstance(self.model.estimator, RegressionResultsWrapper):
            leverage = self.model.estimator._results.get_influence().hat_matrix_diag
        else:
            raise ValueError(
                "Leverage/influence statistics are not available for regularized models. "
                f"The statsmodels output type is {type(self.model.estimator)}."
            )

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
        self.plot_obs_vs_pred(show_outliers=show_outliers, ax=axs[0][0])
        self.plot_residuals_vs_fitted(show_outliers=show_outliers, ax=axs[0][1])
        if self._is_train and isinstance(self.model.estimator, OLSResults):
            self.plot_residuals_vs_leverage(show_outliers=show_outliers, ax=axs[1][0])
        else:
            self.plot_scale_location(show_outliers=show_outliers, ax=axs[1][0])
        self.plot_qq(show_outliers=show_outliers, ax=axs[1][1])
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        decrease_font_sizes_axs(axs, 2, 2, 0)
        plt.close(fig)
        return fig

    def set_outlier_threshold(self, threshold: float) -> "_SingleDatasetOLSReport":
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

        Returns
        ----------
        pd.DataFrame
        """
        return self.scorer.stats_df().astype(float).round(print_options._n_decimals)

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


class OLSReport:
    """OLSReport.
    Fits the model based on provided DataHandler.
    Contains methods for generating regression-relevant diagnostic
    plots and tables for a single linear regression model.
    """

    def __init__(
        self,
        model: OLSLinearModel,
        datahandler: DataHandler,
        target: str,
        predictors: list[str],
        dataemitter: DataEmitter | None = None,
    ):
        """OLSReport.
        Fits the model based on provided DataHandler.
        Contains methods for generating regression-relevant diagnostic
        plots and tables for a single linear regression model.

        Parameters
        ----------
        model : OLSModel

        datahandler : DataHandler
            The DataHandler object that contains the data.

        target : str
            The name of the target variable.

        predictors : list[str]
            The names of the predictor variables.

        dataemitter : DataEmitter
            Default: None. The DataEmitter object that emits the data.
            Optionally you can initialize the report with a DataEmitter object
            instead of a DataHandler object. If not None, will ignore the
            values of target and predictors.
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
        self._train_report = _SingleDatasetOLSReport(model, "train")
        self._test_report = _SingleDatasetOLSReport(model, "test")

    def train_report(self) -> _SingleDatasetOLSReport:
        """Returns an SingleDatasetLinRegReport object for the train dataset

        Returns
        -------
        SingleDatasetLinRegReport
        """
        return self._train_report

    def test_report(self) -> _SingleDatasetOLSReport:
        """Returns an SingleDatasetLinRegReport object for the test dataset

        Returns
        -------
        SingleDatasetLinRegReport
        """
        return self._test_report

    def model(self) -> OLSLinearModel:
        """Returns the fitted OLSLinearModel object.

        Returns
        -------
        OLSLinearModel
        """
        return self._model

    def metrics(self, dataset: Literal["train", "test", "both"]) -> pd.DataFrame:
        """Returns a DataFrame containing the goodness-of-fit statistics
        for the model.

        Parameters
        ----------
        dataset : Literal['train', 'test', 'both']
            The dataset to compute the metrics for.

        Returns
        -------
        pd.DataFrame
        """
        if dataset == "train":
            return self._train_report.metrics()
        elif dataset == "test":
            return self._test_report.metrics()
        elif dataset == "both":
            test_metrics = self._test_report.metrics()  # one column w/ model name
            train_metrics = self._train_report.metrics()  # one column w/ model name
            # stack the two DataFrames on top of each other
            # add an outermost index level to differentiate between the two datasets
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
    ) -> "OLSReport":
        """Performs stepwise selection. Returns a new
        OLSReport object with the reduced model.

        Parameters
        ----------
        direction : Literal["both", "backward", "forward"]
            Default: 'backward'. The direction of the stepwise selection.

        criteria : Literal["aic", "bic"]
            Default: 'aic'. The criteria to use for selecting the best model.

        kept_vars : list[str]
            Default: None. The variables that should be kept in the model.
            If None, defaults to an empty list.

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
        OLSReport
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

        return OLSReport(
            OLSLinearModel(
                alpha=self._model.alpha,
                l1_weight=self._model.l1_weight,
                name=self._model._name + f" (reduced, direction={direction})",
            ),
            self._datahandler,  # only used for y var scaler
            self._target,  # ignored
            selected_vars,  # ignored
            new_emitter,
        )

    def test_lr(self, alternative_report: "OLSReport") -> StatisticalTestReport:
        """Performs a likelihood ratio test to compare an alternative
        OLSLinearModel. Returns an object of class StatisticalTestReport
        describing the results.

        Parameters
        ----------
        alternative_report : OLSReport
            The report of an alternative OLSLinearModel. The alternative
            model must be a nested version of the current model or vice-versa.

        Returns
        -------
        StatisticalTestReport
        """
        if not isinstance(self._model.estimator, RegressionResultsWrapper):
            raise ValueError(
                "Partial F-tests are not available for regularized models. "
                f"The model type is {type(self._model.estimator)}."
            )
        # Determine which report is the reduced model

        # Get the models from each report
        original_model = self._train_report.model.estimator
        alternative_model = alternative_report.train_report().model.estimator

        # Get the number of predictors for each model
        num_predictors_orig = len(self._train_report._X_eval_df.columns)
        num_predictors_alternative = len(
            alternative_report.train_report()._X_eval_df.columns
        )

        if num_predictors_orig > num_predictors_alternative:
            full_model = original_model
            reduced_model = alternative_model
        elif num_predictors_orig < num_predictors_alternative:
            full_model = alternative_model
            reduced_model = original_model
        else:
            # Raise an error if the number of predictors are the same
            raise ValueError("One model must be a reduced version of the other")

        # Raise ValueError if one set of predictors is not a subset of the other
        orig_var_set = set(self._train_report._X_eval_df.columns)
        alt_var_set = set(alternative_report.train_report()._X_eval_df.columns)

        if not (orig_var_set < alt_var_set or orig_var_set > alt_var_set):
            raise ValueError("One model must be a reduced version of the other")

        # Extract the results of the test and temporarily suppress warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            lr_stat, p_value, dr_diff = full_model.compare_lr_test(reduced_model)

        # Initialize and return an object of class StatisticalTestReport
        lr_result = StatisticalTestReport(
            description="Likelihood Ratio Test",
            statistic=lr_stat,
            pval=p_value,
            degfree=dr_diff,
            statistic_description="Chi-square",
            null_hypothesis_description="The full model does not fit the "
            "data significantly better than the reduced model",
            alternative_hypothesis_description="The full model fits the "
            "data signficantly better than the reduced model",
            assumptions_description="The data must be homoscedastic and "
            "uncorrelated",
        )

        return lr_result

    def test_partialf(self, alternative_report: "OLSReport") -> StatisticalTestReport:
        """Performs a partial F-test to compare an alternative OLSLinearModel.
        Returns an object of class StatisticalTestReport describing the results.

        Parameters
        ----------
        alternative_report : OLSReport
            The report of an alternative OLSLinearModel. The alternative
            model must be a nested version of the current model or vice-versa.

        Returns
        -------
        StatisticalTestReport
        """
        if not isinstance(self._model.estimator, RegressionResultsWrapper):
            raise ValueError(
                "Partial F-tests are not available for regularized models."
            )
        # Determine which report is the reduced model

        # Get the models from each report
        original_model = self._train_report.model.estimator
        alternative_model = alternative_report.train_report().model.estimator

        # Get the number of predictors for each model
        num_predictors_orig = len(self._train_report._X_eval_df.columns)
        num_predictors_alternative = len(
            alternative_report.train_report()._X_eval_df.columns
        )

        if num_predictors_orig > num_predictors_alternative:
            full_model = original_model
            reduced_model = alternative_model
        elif num_predictors_orig < num_predictors_alternative:
            full_model = alternative_model
            reduced_model = original_model
        else:
            # Raise an error if the number of predictors are the same
            raise ValueError("One model must be a reduced version of the other")

        # Raise ValueError if one set of predictors is not a subset of the other
        orig_var_set = set(self._train_report._X_eval_df.columns)
        alt_var_set = set(alternative_report.train_report()._X_eval_df.columns)

        if not (orig_var_set < alt_var_set or orig_var_set > alt_var_set):
            raise ValueError("One model must be a reduced version of the other")

        # Extract the results of the test and suppress warnings temporarily
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            f_value, p_value, dr_diff = full_model.compare_f_test(reduced_model)

        # Initialize and return an object of class StatisticalTestReport
        partial_f_result = StatisticalTestReport(
            description="Partial F-Test",
            statistic=f_value,
            pval=p_value,
            degfree=dr_diff,
            statistic_description="F-statistic",
            null_hypothesis_description="The coefficients of the additional "
            "predictors are all zero",
            alternative_hypothesis_description="At least one of the "
            "coefficients of the additional predictors is not zero",
            assumptions_description="The data must be homoscedastic and "
            "have no autocorrelation",
        )

        return partial_f_result

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

    def plot_obs_vs_pred(
        self,
        dataset: Literal["train", "test"],
        show_outliers: bool = True,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Plots a scatter plot of the true and predicted y values.

        Parameters
        ----------
        dataset : Literal['train', 'test']
            The dataset to generate the plot for.

        show_outliers : bool
            Default: True.
            If True, then the outliers calculated using standard errors will be
            shown in red.

        figsize : tuple[float, float]
            Default: (5.0,5.0). Sets the size of the resulting graph.

        ax : plt.Axes
            Default: None.

        Returns
        -------
        - Figure
        """
        if dataset == "train":
            return self._train_report.plot_obs_vs_pred(
                show_outliers=show_outliers, figsize=figsize, ax=ax
            )
        else:
            return self._test_report.plot_obs_vs_pred(
                show_outliers=show_outliers, figsize=figsize, ax=ax
            )

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

    def plot_residuals_vs_var(
        self,
        predictor: str,
        dataset: Literal["train", "test"],
        standardized: bool = False,
        show_outliers: bool = False,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a figure that is a residuals vs fitted (y_pred) plot.

        Parameters
        ----------
        predictor : str
            The predictor variable whose values should be plotted on the x-axis.

        dataset : Literal['train', 'test']
            The dataset to generate the plot for.

        standardized : bool
            Default: False. If True, standardizes the residuals.

        show_outliers : bool
            Default: False. If True, plots the outliers in red.

        figsize : tuple[float, float]
            Default: (5.0, 5.0). Determines the size of the returned figure.

        ax : plt.Axes
            Default: None.

        Returns
        -------
        plt.Figure
        """
        if dataset == "train":
            return self._train_report.plot_residuals_vs_var(
                predictor=predictor,
                standardized=standardized,
                show_outliers=show_outliers,
                figsize=figsize,
                ax=ax,
            )
        elif dataset == "test":
            return self._test_report.plot_residuals_vs_var(
                predictor=predictor,
                standardized=standardized,
                show_outliers=show_outliers,
                figsize=figsize,
                ax=ax,
            )
        else:
            raise ValueError('The dataset must be either "train" or "test".')

    def plot_residuals_hist(
        self,
        dataset: Literal["train", "test"],
        standardized: bool = False,
        density: bool = False,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a figure that is a histogram of the residuals.

        Parameters
        ----------
        dataset : Literal['train', 'test']
            The dataset to generate the plot for.

        standardized : bool
            Default: False. If True, standardizes the residuals.

        density : bool
            Default: False. If True, plots density rather than frequency.

        figsize : tuple[float, float]
            Default: (5.0, 5.0). Determines the size of the returned figure.

        ax : plt.Axes
            Default: None.

        Returns
        -------
        plt.Figure
        """
        if dataset == "train":
            return self._train_report.plot_residuals_hist(
                standardized=standardized, density=density, figsize=figsize, ax=ax
            )
        elif dataset == "test":
            return self._test_report.plot_residuals_hist(
                standardized=standardized, density=density, figsize=figsize, ax=ax
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

    def set_outlier_threshold(self, threshold: float) -> "OLSReport":
        """Standardized residuals threshold for outlier identification.
        Recomputes the outliers.

        Parameters
        ----------
        threshold : float
            Default: 2. Must be a nonnegative value.

        Returns
        -------
        OLSReport
            Returns self for method chaining.
        """
        self._train_report.set_outlier_threshold(threshold=threshold)
        self._test_report.set_outlier_threshold(threshold=threshold)
        return self

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
        """Returns the coefficients of the model.

        Parameters
        ----------
        format : Literal["coef(se)|pval", "coef|se|pval", "coef(ci)|pval",
                        "coef|ci_low|ci_high|pval"]
            Default: 'coef(se)|pval'.

        Returns
        -------
        pd.DataFrame
        """
        return self._model.coefs(format)

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
        """Returns the JSON serializable data stored in the report as a dictionary.

        Returns
        -------
        dict
        """
        return {
            "coefficients": self.coefs("coef(se)|pval").to_dict("index"),
            "train_metrics": self.metrics("train").to_dict("index"),
            "test_metrics": self.metrics("test").to_dict("index"),
        }

    def __str__(self) -> str:
        max_width = print_options._max_line_width
        n_dec = print_options._n_decimals

        top_divider = color_text("=" * max_width, "none") + "\n"
        bottom_divider = "\n" + color_text("=" * max_width, "none")
        divider = "\n" + color_text("-" * max_width, "none") + "\n"
        divider_invisible = "\n" + " " * max_width + "\n"

        if self._model.alpha == 0:
            title_message = bold_text("Ordinary Least Squares Regression Report")
        else:
            if self._model.l1_weight == 0:
                title_message = bold_text(
                    f"Ridge Regression Report (alpha={self._model.alpha})"
                )
            elif self._model.l1_weight == 1:
                title_message = bold_text(
                    f"Lasso Regression Report (alpha={self._model.alpha})"
                )
            else:
                title_message = bold_text(
                    f"Elastic Net Regression Report (alpha={self._model.alpha}, "
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
                "R2:       "
                + color_text(
                    str(np.round(self.metrics("train").at["r2", mstr], n_dec)), "yellow"
                ),
                "R2:       "
                + color_text(
                    str(np.round(self.metrics("test").at["r2", mstr], n_dec)), "yellow"
                ),
                total_len=max_width - 2,
            ),
            initial_indent=4,
        )
        metrics_message += "\n"
        metrics_message += fill_ignore_format(
            format_two_column(
                "Adj. R2:  "
                + color_text(
                    str(np.round(self.metrics("train").at["adjr2", mstr], n_dec)),
                    "yellow",
                ),
                "Adj. R2:  "
                + color_text(
                    str(np.round(self.metrics("test").at["adjr2", mstr], n_dec)),
                    "yellow",
                ),
                total_len=max_width - 2,
            ),
            initial_indent=4,
        )
        metrics_message += "\n"
        metrics_message += fill_ignore_format(
            format_two_column(
                "RMSE:     "
                + color_text(
                    str(np.round(self.metrics("train").at["rmse", mstr], n_dec)),
                    "yellow",
                ),
                "RMSE:     "
                + color_text(
                    str(np.round(self.metrics("test").at["rmse", mstr], n_dec)),
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

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))
