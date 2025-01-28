import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from typing import Literal
from ..metrics.classification_scoring import ClassificationBinaryScorer
from ..data.datahandler import DataEmitter
from ..utils import ensure_arg_list_uniqueness, is_numerical
from ..display.print_utils import suppress_std_output, suppress_print_output
from .lmutils.score import score_model
from ..display.print_options import print_options
from ..ml.predict.classification.thresholding_utils import (
    select_optimal_threshold_binary,
    predict_with_threshold_binary,
)


class LogitLinearModel:
    """Statsmodels Logit wrapper."""

    def __init__(
        self,
        alpha: float = 0.0,
        l1_weight: float = 0.0,
        threshold_strategy: Literal["f1", "roc"] | None = "roc",
        name: str = "Logit Linear Model",
    ):
        """
        Initializes a LogitLinearModel object.

        Parameters
        ----------
        alpha : float
            Default: 0. Regularization strength. Must be a positive float.

        l1_weight : float
            Default: 0. The weight of the L1 penalty. Must be a float between 0 and 1.

        name : str
            Default: 'Logit Linear Model'. The name of the model.
        """
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if l1_weight < 0 or l1_weight > 1:
            raise ValueError("l1_weight must be between 0 and 1")

        self.alpha = alpha
        self.l1_weight = l1_weight

        self.estimator = None
        self._name = name
        self._label_encoder = None
        self._threshold_strategy = threshold_strategy

    def specify_data(self, dataemitter: DataEmitter):
        """Adds a DataEmitter object to the model.

        Parameters
        ----------
        dataemitter : DataEmitter
            The DataEmitter containing all the data.
        """
        self._dataemitter = dataemitter

    def fit(self, max_iter: int | None = None, verbose: bool = True):
        """Fits the model based on the data specified."""

        # Emit all data
        X_train, y_train = self._dataemitter.emit_train_Xy(
            dropfirst=True, verbose=verbose
        )
        X_test, y_test = self._dataemitter.emit_test_Xy(dropfirst=True, verbose=verbose)

        self._predictors = X_train.columns.to_list()
        self._n_predictors = len(self._predictors)
        self._n_test = len(X_test)
        self._n_train = len(X_train)

        # we force the constant to be included
        X_train = sm.add_constant(X_train, has_constant="add")
        X_test = sm.add_constant(X_test, has_constant="add")

        y_levels = y_train.unique()
        if y_levels.size != 2:
            raise ValueError("Target variable must have 2 levels")

        y_test_levels = y_test.unique()
        if y_test_levels.size > 2:
            raise ValueError(
                "Target variable in test set detected to have more than 2 levels"
            )

        # we allow y_train to be categorical, i.e. we encode it with a label encoder
        self._y_label_order = None
        if not is_numerical(y_train):
            self._label_encoder = LabelEncoder()
            y_train = self._label_encoder.fit_transform(y_train)
            self._y_label_order = self._label_encoder.classes_
            y_test = self._label_encoder.transform(y_test)

        with suppress_std_output():
            if self.alpha == 0:
                if max_iter is None:
                    max_iter = 50
                self.estimator = sm.Logit(y_train, X_train).fit(
                    cov_type="HC3", maxiter=max_iter
                )
            else:
                if max_iter is None:
                    max_iter = "defined_by_method"
                self.estimator = sm.Logit(y_train, X_train).fit_regularized(
                    alpha=self.alpha, L1_wt=self.l1_weight, maxiter=max_iter
                )

        y_pred_train: np.ndarray = self.estimator.predict(exog=X_train).to_numpy()

        self._threshold = select_optimal_threshold_binary(
            y_true=y_train, y_pred_score=y_pred_train, metric=self._threshold_strategy
        )

        y_pred_train_binary = predict_with_threshold_binary(
            y_pred_score=y_pred_train, threshold=self._threshold
        )

        self.train_scorer = ClassificationBinaryScorer(
            y_pred=y_pred_train_binary,
            y_true=y_train,
            pos_label=self._y_label_order[1] if self._y_label_order is not None else 1,
            y_pred_score=y_pred_train,
            name=self._name,
        )

        y_pred_test = self.estimator.predict(X_test).to_numpy()

        y_pred_test_binary = predict_with_threshold_binary(
            y_pred_score=y_pred_test, threshold=self._threshold
        )

        y_pred_test_reshaped = y_pred_test.reshape(-1, 1)

        self.test_scorer = ClassificationBinaryScorer(
            y_pred=y_pred_test_binary,
            y_true=y_test,
            pos_label=self._y_label_order[1] if self._y_label_order is not None else 1,
            y_pred_score=np.hstack([1 - y_pred_test_reshaped, y_pred_test_reshaped]),
            name=self._name,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts the target variable for the given data."""

        # ensure X contains the correct predictors
        if not set(X.columns) == set(self._dataemitter.X_vars()):
            raise ValueError("X must contain the same predictors as the training data")

        X = sm.add_constant(X, has_constant="add")
        y_pred = self.estimator.predict(X).to_numpy()
        return predict_with_threshold_binary(
            y_pred_score=y_pred, threshold=self._threshold
        )

    @ensure_arg_list_uniqueness()
    def step(
        self,
        direction: Literal["both", "backward", "forward"] = "backward",
        criteria: Literal["aic", "bic"] = "aic",
        kept_vars: list[str] | None = None,
        all_vars: list[str] | None = None,
        start_vars: list[str] | None = None,
        max_steps: int = 100,
    ) -> list[str]:
        """This method implements stepwise selection for identifying important
        features. If the direction is set to forward, the algorithm will start
        with no selected variables and will at each time step add every
        left-out feature to the model separately. the left-out feature
        that results in the best improvement in the metric (aic or bic) will
        be selected as an important feature. This happens until all variables
        are added or adding a left-out variable does not improve the metric
        of choice.

        If the direction is set to backward, the algorithm will start with all
        variables selected and will at each time step remove each included
        variable separately. The variable that results in the best improvement
        in the metric when removed from the model will be removed from the
        list of selected features.

        Categorical variables will either be included or excluded as a whole.

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
            Default: None. The variables to start the bidirectional stepwise selection with.
            Ignored if direction is not 'both'. If direction is 'both' and
            start_vars is None, then the starting variables are the kept_vars.

        max_steps : int
            Default: 100. The maximum number of steps to take.

        Returns
        -------
        list[str]
            The subset of predictors that are most likely to be significant.
        """
        if max_steps <= 0:
            raise ValueError("max_steps cannot be non-positive")

        # set upper to all possible variables if nothing is specified
        local_dataemitter = self._dataemitter.copy()
        if all_vars is None:
            all_vars = local_dataemitter.X_vars()
        if kept_vars is None:
            kept_vars = []

        # ensure that kept vars are in all vars
        for var in kept_vars:
            if var not in all_vars:
                raise ValueError(f"{var} is not in all_vars")

        # set our current variables to our starting list
        if direction == "forward":
            included_vars = kept_vars.copy()
        elif direction == "backward":
            included_vars = all_vars.copy()
        elif direction == "both":
            if start_vars is None:
                included_vars = kept_vars.copy()
            else:
                included_vars = start_vars.copy()
        else:
            raise ValueError("direction must be 'both', 'backward', or 'forward'")

        with suppress_std_output(), suppress_print_output():
            # set our starting score and best models
            current_score = score_model(
                local_dataemitter,
                included_vars,
                model="logit",
                alpha=self.alpha,
                l1_weight=self.l1_weight,
                metric=criteria,
                y_label_encoder=self._label_encoder,
            )
            current_step = 0

            while current_step < max_steps:
                # Forward step
                if direction == "forward":
                    excluded = list(set(all_vars) - set(included_vars))

                    best_score = current_score
                    var_to_add = None
                    for new_var in excluded:
                        candidate_features = included_vars + [new_var]
                        score = score_model(
                            local_dataemitter,
                            candidate_features,
                            model="logit",
                            alpha=self.alpha,
                            l1_weight=self.l1_weight,
                            metric=criteria,
                            y_label_encoder=self._label_encoder,
                        )
                        if score < best_score:
                            best_score = score
                            var_to_add = new_var

                    # If we didn't find a variable to add (score is not better), break
                    if var_to_add is None:
                        break

                    included_vars.append(var_to_add)

                # Backward step
                elif direction == "backward":
                    if len(included_vars) <= len(kept_vars):
                        break

                    best_score = current_score
                    var_to_remove = None

                    for candidate in included_vars:
                        if candidate in kept_vars:
                            continue

                        candidate_features = included_vars.copy()
                        candidate_features.remove(candidate)
                        score = score_model(
                            local_dataemitter,
                            candidate_features,
                            model="logit",
                            alpha=self.alpha,
                            l1_weight=self.l1_weight,
                            metric=criteria,
                            y_label_encoder=self._label_encoder,
                        )
                        if score < best_score:
                            best_score = score
                            var_to_remove = candidate

                    if var_to_remove is None:
                        break

                    included_vars.remove(var_to_remove)

                elif direction == "both":
                    excluded = list(set(all_vars) - set(included_vars))

                    best_score = current_score

                    best_forward_score = current_score
                    var_to_add = None
                    for new_var in excluded:
                        candidate_features = included_vars + [new_var]
                        score = score_model(
                            local_dataemitter,
                            candidate_features,
                            model="logit",
                            alpha=self.alpha,
                            l1_weight=self.l1_weight,
                            metric=criteria,
                            y_label_encoder=self._label_encoder,
                        )
                        if score < best_forward_score:
                            best_forward_score = score
                            var_to_add = new_var

                    best_backward_score = current_score
                    var_to_remove = None

                    for candidate in included_vars:
                        if candidate in kept_vars:
                            continue

                        candidate_features = included_vars.copy()
                        candidate_features.remove(candidate)
                        score = score_model(
                            local_dataemitter,
                            candidate_features,
                            model="logit",
                            alpha=self.alpha,
                            l1_weight=self.l1_weight,
                            metric=criteria,
                            y_label_encoder=self._label_encoder,
                        )
                        if score < best_backward_score:
                            best_backward_score = score
                            var_to_remove = candidate

                    if best_forward_score < best_backward_score:
                        if var_to_add is None:
                            break
                        included_vars.append(var_to_add)
                        best_score = best_forward_score
                    else:
                        if var_to_remove is None:
                            break
                        included_vars.remove(var_to_remove)
                        best_score = best_backward_score

                current_score = best_score
                current_step += 1

        return included_vars

    def coefs(
        self,
        format: Literal[
            "coef(se)|pval", "coef|se|pval", "coef(ci)|pval", "coef|ci_low|ci_high|pval"
        ] = "coef(se)|pval",
    ) -> pd.DataFrame:
        """Returns a DataFrame containing the coefficients, standard errors, 
        and p-values. The standard errors and p-values are heteroskedasticity-
        robust. Confidence intervals are reported at 95% confidence level if 
        applicable.
        
        Parameters
        ----------
        format : Literal["coef(se)|pval", "coef|se|pval", "coef(ci)|pval",\
                            "coef|ci_low|ci_high|pval"]
            Default: "coef(se)|pval". The format of the output DataFrame.
        """
        params = self.estimator.params
        std_err = self.estimator.bse
        p_values = self.estimator.pvalues

        two_stdevs = 1.959963984540054

        ci_low = params - two_stdevs * std_err
        ci_high = params + two_stdevs * std_err

        output_df = pd.DataFrame(
            {
                "coef": np.round(params, print_options._n_decimals),
                "se": np.round(std_err, print_options._n_decimals),
                "pval": np.round(p_values, print_options._n_decimals),
                "ci_low": np.round(ci_low, print_options._n_decimals),
                "ci_high": np.round(ci_high, print_options._n_decimals),
            }
        )
        if format == "coef(se)|pval":
            output_df["coef(se)"] = output_df.apply(
                lambda row: f"{row['coef']} ({row['se']})", axis=1
            )
            output_df = output_df[["coef(se)", "pval"]]
            output_df = output_df.rename(
                columns={"coef(se)": "Estimate (Std. Error)", "pval": "p-value"}
            )
        elif format == "coef|se|pval":
            output_df = output_df[["coef", "se", "pval"]]
            output_df = output_df.rename(
                columns={"coef": "Estimate", "se": "Std. Error", "pval": "p-value"}
            )
        elif format == "coef(ci)|pval":
            output_df["ci_str"] = output_df["coef"] + two_stdevs * output_df["se"]
            output_df["coef(ci)"] = output_df.apply(
                lambda row: f"{row['coef']} ({row['ci_low']}, {row['ci_high']})", axis=1
            )
            output_df = output_df[["coef(ci)", "pval"]]
            output_df = output_df.rename(
                columns={"coef(ci)": "Estimate (95% CI)", "pval": "p-value"}
            )
        elif format == "coef|ci_low|ci_high|pval":
            output_df = output_df.rename(
                columns={
                    "coef": "Estimate",
                    "ci_low": "CI Lower Bound",
                    "ci_high": "CI Upper Bound",
                    "pval": "p-value",
                }
            )
        else:
            raise ValueError("Invalid format")
        return output_df

    def positive_class(self) -> str:
        """Returns the positive class label name as a string. \
        That is, the class label that is being predicted by the model."""
        return self._y_label_order[1] if self._y_label_order is not None else "1"

    def __str__(self):
        return self._name
