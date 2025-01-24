import pandas as pd
import numpy as np
from typing import Literal
from ...data.datahandler import DataHandler, DataEmitter
from ..mnlogit import MNLogitLinearModel
from ...display.print_options import print_options
from ...display.print_utils import (
    color_text,
    bold_text,
    suppress_print_output,
    list_to_string,
    fill_ignore_format,
    format_two_column,
    print_wrapped,
)


class _SingleDatasetMNLogitReport:
    """Class for generating regression-relevant diagnostic
    plots and tables for a MNLogit model.
    """

    def __init__(self, model: MNLogitLinearModel, dataset: Literal["train", "test"]):
        """
        Initializes a _SingleDatasetMNLogitReport object.

        Parameters
        ----------
        model : MNLogitLinearModel
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
        self._y_pred = self.scorer._y_pred
        self._y_true = self.scorer._y_true

        if np.isnan(self._y_pred_score).sum() > 0:
            raise ValueError(
                "NaNs found in predictions. Please try refitting with regularization."
            )

        self._include_text = False

    def metrics(self) -> pd.DataFrame:
        """Returns a DataFrame containing the goodness-of-fit statistics
        for the model.

        Parameters
        ----------
        pd.DataFrame
        """
        return self.scorer.stats_df()


class MNLogitReport:
    """MNLogitReport.
    Fits the model based on provided DataHandler.
    Wraps train and test SingleDatasetMNLogitReport objects.
    """

    def __init__(
        self,
        model: MNLogitLinearModel,
        datahandler: DataHandler,
        target: str,
        predictors: list[str],
        dataemitter: DataEmitter | None = None,
    ):
        """LogitRegressionReport.
        Fits the model based on provided DataHandler.
        Wraps train and test SingleDatasetMNLogitReport objects.

        Parameters
        ----------
        model : MNLogitLinearModel

        datahandler : DataHandler
            The DataHandler object that contains the data.

        target : str
            The name of the dependent variable.

        predictors : list[str]
            The names of the independent variables.
        """
        max_iter = 200

        self._model = model
        self._datahandler = datahandler
        if dataemitter is not None:
            self._dataemitter = dataemitter
        else:
            self._dataemitter = self._datahandler.train_test_emitter(target, predictors)
        self._model.specify_data(self._dataemitter)
        self._model.fit(max_iter=max_iter)
        self._target = target
        self._predictors = predictors
        self._train_report = _SingleDatasetMNLogitReport(model, "train")
        self._test_report = _SingleDatasetMNLogitReport(model, "test")

    def step(
        self,
        direction: Literal["both", "backward", "forward"] = "backward",
        criteria: Literal["aic", "bic"] = "aic",
        kept_vars: list[str] | None = None,
        all_vars: list[str] | None = None,
        start_vars: list[str] | None = None,
        max_steps: int = 100,
    ) -> "MNLogitReport":
        """Performs stepwise selection on the model. Returns a new
        MNLogitReport object with the updated model.

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
        MNLogitReport
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

        return MNLogitReport(
            MNLogitLinearModel(
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

    def model(self) -> MNLogitLinearModel:
        """Returns the fitted MNLogitLinearModel object.

        Returns
        -------
        MNLogitLinearModel
        """
        return self._model

    def metrics(self, dataset: Literal["train", "test", "both"]) -> pd.DataFrame:
        """Returns a DataFrame containing the goodness-of-fit statistics
        for the model.

        Parameters
        ----------
        dataset : Literal['train', 'test', 'both']
            The dataset to generate the report for.

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

    def _to_dict(self) -> dict:
        return {
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

        title_message = bold_text("Multinomial Logistic Regression Report")

        target_var = "'" + self._target + "'"
        target_message = f"{bold_text('Target variable:')}\n"
        target_message += fill_ignore_format(
            color_text(target_var, "purple"),
            width=max_width,
            initial_indent=2,
            subsequent_indent=2,
        )

        predictors_message = (
            f"{bold_text(f'Predictor variables ({self._model._n_predictors}):')}\n"
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
                "F1:   "
                + color_text(
                    str(np.round(self.metrics("train").at["f1", mstr], n_dec)), "yellow"
                ),
                "F1:   "
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
                "Acc:  "
                + color_text(
                    str(np.round(self.metrics("train").at["accuracy", mstr], n_dec)),
                    "yellow",
                ),
                "Acc:  "
                + color_text(
                    str(np.round(self.metrics("test").at["accuracy", mstr], n_dec)),
                    "yellow",
                ),
                total_len=max_width - 2,
            ),
            initial_indent=4,
        )
        final_message = (
            top_divider
            + title_message
            + divider
            + target_message
            + divider_invisible
            + predictors_message
            + divider
            + metrics_message
            + bottom_divider
        )

        return final_message

    def _repr_pretty_(self, p, cycle) -> str:
        p.text(str(self))
