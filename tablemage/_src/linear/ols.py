import statsmodels.api as sm
from typing import Literal
import pandas as pd
import numpy as np
from scipy.stats import t
from ..metrics.regression_scoring import RegressionScorer
from ..data.datahandler import DataEmitter
from ..utils import ensure_arg_list_uniqueness
from ..display.print_utils import suppress_print_output
from ..display.print_options import print_options
from .lmutils.score import score_model


def bootstrap_coefs_and_pvals(
    model: sm.regression.linear_model.OLS,
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = 0.0,
    l1_weight: float = 0.0,
    n_bootstraps: int = 5000,
    random_state: int = 42,
):
    """
    Perform bootstrapping to estimate coefficient standard errors and p-values
    for a statsmodels regularized regression model.

    Parameters
    ----------
    model : statsmodels.regression.linear_model.OLS
        The regularized model to fit (e.g., sm.OLS(y, X).fit_regularized()).

    X : pd.DataFrame
        Feature matrix.

    y : pd.Series
        Target vector.

    alpha : float, default=0.0

    l1_weight : float, default=0.0

    n_bootstraps : int, default=1000
        Number of bootstrap samples.

    random_state : int, default=None
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing coefficients, standard errors, and p-values.
    """
    np.random.seed(random_state)
    coefs = []

    base_model = model.fit_regularized(alpha=alpha, L1_wt=l1_weight)
    base_coefs = base_model.params

    for _ in range(n_bootstraps):
        indices = np.random.choice(len(y), len(y), replace=True)
        X_bootstrap = X.iloc[indices] if isinstance(X, pd.DataFrame) else X[indices]
        y_bootstrap = y.iloc[indices] if isinstance(y, pd.Series) else y[indices]

        bootstrap_model = sm.OLS(y_bootstrap, X_bootstrap).fit_regularized(
            alpha=alpha, L1_wt=l1_weight
        )
        coefs.append(bootstrap_model.params)

    coefs = np.array(coefs)
    standard_errors = np.std(coefs, axis=0)

    t_stats = base_coefs / standard_errors
    p_values = 2 * (1 - t.cdf(np.abs(t_stats), df=len(y) - X.shape[1]))

    results = pd.DataFrame(
        {
            "Estimate": base_coefs,
            "Std Error": standard_errors,
            "t-stat": t_stats,
            "p-value": p_values,
        },
        index=X.columns,
    )

    return results


class OLSLinearModel:
    """Statsmodels OLS wrapper."""

    def __init__(
        self,
        alpha: float = 0.0,
        l1_weight: float = 0.0,
        name: str = "OLS Linear Model",
    ):
        """
        Initializes a OLSLinearModel object.

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

    def specify_data(self, dataemitter: DataEmitter):
        """Adds a DataEmitter object to the model.

        Parameters
        ----------
        dataemitter : DataEmitter
            The DataEmitter containing all the data.
        """
        self._dataemitter = dataemitter

    def fit(self, verbose: bool = True):
        """Fits the model based on the data specified."""
        y_scaler = self._dataemitter.y_scaler()

        X_train, y_train = self._dataemitter.emit_train_Xy(
            dropfirst=True, verbose=verbose
        )
        X_test, y_test = self._dataemitter.emit_test_Xy(dropfirst=True, verbose=verbose)

        self._predictors = X_train.columns.to_list()
        self._n_predictors = len(self._predictors)
        self._n_train = len(X_train)
        self._n_test = len(X_test)

        # we force the constant to be included
        X_train = sm.add_constant(X_train, has_constant="add")
        X_test = sm.add_constant(X_test, has_constant="add")
        self._X_train_orig = X_train
        self._y_train_orig = y_train

        if self.alpha == 0:
            self.estimator = sm.OLS(y_train, X_train).fit(cov_type="HC3")
        else:
            self.estimator = sm.OLS(y_train, X_train).fit_regularized(
                alpha=self.alpha, L1_wt=self.l1_weight
            )

        y_pred_train = self.estimator.predict(X_train).to_numpy()
        if y_scaler is not None:
            y_pred_train = y_scaler.inverse_transform(y_pred_train)
            y_train = y_scaler.inverse_transform(y_train)

        self.train_scorer = RegressionScorer(
            y_pred=y_pred_train,
            y_true=y_train.to_numpy(),
            n_predictors=self._n_predictors,
            name=self._name,
        )

        if X_train.shape[1] == self._n_predictors:
            exception_vars = set(X_train.columns).symmetric_difference(
                set(X_test.columns)
            )
            raise RuntimeError(
                "Mismatched train/test predictors. "
                f"{exception_vars} not found in both train and test."
            )

        y_pred_test = self.estimator.predict(X_test).to_numpy()
        if y_scaler is not None:
            y_pred_test = y_scaler.inverse_transform(y_pred_test)
            y_test = y_scaler.inverse_transform(y_test)

        self.test_scorer = RegressionScorer(
            y_pred=y_pred_test,
            y_true=y_test.to_numpy(),
            n_predictors=self._n_predictors,
            name=self._name,
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
            raise ValueError("max_steps cannot be negative")

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
            raise ValueError("direction must be 'forward', 'backward', or 'both'")

        with suppress_print_output():
            # set our starting score and best models
            current_score = score_model(
                local_dataemitter,
                included_vars,
                model="ols",
                alpha=self.alpha,
                l1_weight=self.l1_weight,
                metric=criteria,
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
                            model="ols",
                            alpha=self.alpha,
                            l1_weight=self.l1_weight,
                            metric=criteria,
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
                            model="ols",
                            alpha=self.alpha,
                            l1_weight=self.l1_weight,
                            metric=criteria,
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
                            model="ols",
                            alpha=self.alpha,
                            l1_weight=self.l1_weight,
                            metric=criteria,
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
                            model="ols",
                            alpha=self.alpha,
                            l1_weight=self.l1_weight,
                            metric=criteria,
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

        if self.alpha == 0:
            std_err = self.estimator.bse
            p_values = self.estimator.pvalues
        else:
            bootstrap_results = bootstrap_coefs_and_pvals(
                sm.OLS(self._y_train_orig, self._X_train_orig),
                self._X_train_orig,
                self._y_train_orig,
            )
            std_err = bootstrap_results["Std Error"]
            p_values = bootstrap_results["p-value"]

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

    def __str__(self):
        return self._name
