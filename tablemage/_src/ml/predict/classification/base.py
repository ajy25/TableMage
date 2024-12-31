import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from typing import Literal
from ....feature_selection import BaseFSC
from ..base_model import BasePredictModel, HyperparameterSearcher
from ....data import DataEmitter
from ....metrics import (
    ClassificationMulticlassScorer,
    ClassificationBinaryScorer,
)
from ....feature_selection import VotingSelectionReport
from ....display.print_utils import (
    print_wrapped,
    color_text,
    quote_and_color,
)
from ....display.print_options import print_options
from ..predict_utils import ColumnSelector
from .thresholding_utils import (
    select_optimal_threshold_binary,
    predict_with_threshold_binary,
    select_optimal_thresholds_multiclass,
    predict_with_thresholds_multiclass,
)


class BaseC(BasePredictModel):
    """BaseC is a class that provides a training and evaluation framework that all
    TableMage classificaiton classes inherit (i.e., all ___C classes are children
    of BaseC). The primary purpose of BaseC is to automate the scoring and
    model selection processes.
    """

    def __init__(self, threshold_strategy: Literal["f1", "roc"] | None = "roc"):
        """Initializes a BaseC object."""
        self._label_encoder = LabelEncoder()
        self._hyperparam_searcher: HyperparameterSearcher = None
        self._best_estimator: BaseEstimator = None
        self._dataemitter = None
        self._dataemitters = None
        self._feature_selectors = None
        self._max_n_features = None
        self._name = "BaseC"
        self._train_scorer = None
        self._cv_scorer = None
        self._test_scorer = None
        self._feature_selection_report = None
        self._predictors = None
        self._is_binary = True
        self._threshold_strategy = threshold_strategy
        self._threshold = None
        self._feature_importance_type_str = "feature importances"

        # By default, the first level is NOT dropped unless binary. For linear models,
        # the first level is dropped to avoid multicollinearity.
        self._dropfirst = False

    def specify_data(
        self,
        dataemitter: DataEmitter | None = None,
        dataemitters: list[DataEmitter] | None = None,
    ):
        """Specifies the DataEmitters for the model fitting process.

        Parameters
        ----------
        dataemitter : DataEmitter | None
            Default: None.
            DataEmitter that contains the data. If not None, re-specifies the
            DataEmitter for the model.

        dataemitters : list[DataEmitter] | None
            Default: None.
            If not None, re-specifies the DataEmitters for nested cross validation.
        """
        if dataemitter is not None:
            self._dataemitter = dataemitter
        if dataemitters is not None:
            self._dataemitters = dataemitters

    def fit(self, verbose: bool = False):
        """Fits and evaluates the model.

        The model fitting process is as follows:

        1. The train data is emitted. This means that the data is preprocessed based on
        user specifications AND necessary automatic preprocessing steps. That is,
        the DataEmitter will automatically drop observations with missing
        entries and encode categorical variables IF NOT SPECIFIED BY USER.
        2. The hyperparameter search is performed. The best estimator is saved and
        evaluated on the train data.
        3. The test data is emitted. Preprocessing steps were previously
        fitted on the train data. The test data is transformed accordingly.
        4. The best estimator determined from the training step
        is evaluated on the test data.

        If cross validation is specified, fold-specific DataEmitters are generated.
        Steps 1-4 are repeated for each fold.

        The fitting process yields three sets of metrics:

        1. The training set metrics.
        2. The cross validation set metrics. *only if cross validation was specified*
            Note that the cross validation metrics are computed on the test set of
            each fold and are therefore a more robust estimate of model performance
            than the test set metrics.
        3. The test set metrics.

        Parameters
        ----------
        verbose : bool
            Default: False. If True, prints progress.
        """
        self._is_binary = True

        if self._dataemitters is None and self._dataemitter is not None:
            X_train_df, y_train_series = self._dataemitter.emit_train_Xy(
                dropfirst=self._dropfirst, verbose=verbose
            )
            y_train = y_train_series.to_numpy()

            if self._feature_selectors is not None:
                self._feature_selection_report = VotingSelectionReport(
                    selectors=self._feature_selectors,
                    dataemitter=self._dataemitter,
                    max_n_features=self._max_n_features,
                    verbose=verbose,
                )
                self._predictors = self._feature_selection_report.top_features()
                X_train = self._feature_selection_report._emit_train_X(
                    dropfirst=self._dropfirst, verbose=verbose
                )
            else:
                self._predictors = X_train_df.columns.to_list()
                X_train = X_train_df

            y_train_encoded = self._label_encoder.fit_transform(y_train)

            if len(self._label_encoder.classes_) > 2:
                self._is_binary = False

            self._hyperparam_searcher.fit(X_train, y_train_encoded, verbose)
            self._best_estimator = self._hyperparam_searcher._best_estimator

            y_pred_encoded = self._best_estimator.predict(X_train)

            if hasattr(self._best_estimator, "predict_proba"):
                y_pred_score = self._best_estimator.predict_proba(X_train)

                if self._is_binary:
                    self._threshold = select_optimal_threshold_binary(
                        y_train_encoded,
                        y_pred_score[:, 1],
                        metric=self._threshold_strategy,
                    )
                    y_pred_encoded = predict_with_threshold_binary(
                        y_pred_score[:, 1], self._threshold
                    )

                    if verbose and self._threshold_strategy is not None:
                        print_wrapped(
                            f"Optimal threshold set for {quote_and_color(self._name)} "
                            f"via {self._threshold_strategy}.",
                            type="PROGRESS",
                        )
                else:
                    self._threshold = select_optimal_thresholds_multiclass(
                        y_train_encoded, y_pred_score, metric=self._threshold_strategy
                    )
                    y_pred_encoded = predict_with_thresholds_multiclass(
                        y_pred_score, thresholds=self._threshold
                    )

                    if verbose and self._threshold_strategy is not None:
                        print_wrapped(
                            f"Optimal thresholds set for {quote_and_color(self._name)} "
                            f"via {self._threshold_strategy}.",
                            type="PROGRESS",
                        )

            elif hasattr(self._best_estimator, "decision_function"):
                y_pred_score = self._best_estimator.decision_function(X_train)

            if self._is_binary:
                self._train_scorer = ClassificationBinaryScorer(
                    y_pred=y_pred_encoded,
                    y_true=y_train_encoded,
                    pos_label=self._label_encoder.classes_[1],
                    y_pred_score=y_pred_score,
                    name=str(self),
                )

            else:
                self._train_scorer = ClassificationMulticlassScorer(
                    y_pred=self._label_encoder.inverse_transform(y_pred_encoded),
                    y_true=y_train,
                    y_pred_score=y_pred_score,
                    y_pred_class_order=self._label_encoder.classes_,
                    name=str(self),
                )

        elif self._dataemitters is not None and self._dataemitter is not None:
            y_preds_encoded = []
            y_trues = []
            y_pred_scores = []

            for emitter in self._dataemitters:
                (
                    X_train_df,
                    y_train_series,
                    X_test_df,
                    y_test_series,
                ) = emitter.emit_train_test_Xy(
                    dropfirst=self._dropfirst, verbose=verbose
                )
                y_train = y_train_series.to_numpy()
                y_train_encoded: np.ndarray = self._label_encoder.fit_transform(y_train)

                if self._feature_selectors is not None:
                    fold_selector = VotingSelectionReport(
                        selectors=self._feature_selectors,
                        dataemitter=emitter,
                        max_n_features=self._max_n_features,
                        verbose=verbose,
                    )
                    X_train = fold_selector._emit_train_X(
                        dropfirst=self._dropfirst, verbose=verbose
                    )
                    X_test = fold_selector._emit_test_X(
                        dropfirst=self._dropfirst, verbose=verbose
                    )
                else:
                    X_train = X_train_df
                    X_test = X_test_df

                if len(self._label_encoder.classes_) > 2:
                    self._is_binary = False

                y_test = y_test_series.to_numpy()

                self._hyperparam_searcher.fit(X_train, y_train_encoded, verbose)
                fold_estimator = self._hyperparam_searcher._best_estimator

                y_trues.append(y_test)

                fold_threshold = None

                if hasattr(fold_estimator, "predict_proba"):
                    y_pred_score_fold = fold_estimator.predict_proba(X_test)
                    if self._is_binary:
                        fold_threshold = select_optimal_threshold_binary(
                            self._label_encoder.transform(y_test),
                            y_pred_score_fold[:, 1],
                            metric=self._threshold_strategy,
                        )
                        y_pred_scores.append(y_pred_score_fold)
                    else:
                        fold_threshold = select_optimal_thresholds_multiclass(
                            self._label_encoder.transform(y_test),
                            y_pred_score_fold,
                            metric=self._threshold_strategy,
                        )
                        y_pred_scores.append(y_pred_score_fold)
                elif hasattr(self._best_estimator, "decision_function"):
                    y_pred_score_fold = fold_estimator.decision_function(X_test)
                    y_pred_scores.append(y_pred_score_fold)

                if self._is_binary:
                    y_pred_encoded = predict_with_threshold_binary(
                        y_pred_score_fold[:, 1], threshold=fold_threshold
                    )
                else:
                    y_pred_encoded = predict_with_thresholds_multiclass(
                        y_pred_score_fold, thresholds=fold_threshold
                    )
                y_preds_encoded.append(y_pred_encoded)

            if len(y_pred_scores) == 0:
                y_pred_scores = None

            if self._is_binary:
                self._cv_scorer = ClassificationBinaryScorer(
                    y_pred=y_preds_encoded,
                    y_true=[self._label_encoder.transform(y) for y in y_trues],
                    pos_label=self._label_encoder.classes_[1],
                    y_pred_score=y_pred_scores,
                    name=str(self),
                )
            else:
                self._cv_scorer = ClassificationMulticlassScorer(
                    y_pred=[
                        self._label_encoder.inverse_transform(y)
                        for y in y_preds_encoded
                    ],
                    y_true=y_trues,
                    y_pred_score=y_pred_scores,
                    y_pred_class_order=self._label_encoder.classes_,
                    name=str(self),
                )

            # refit on all data
            X_train_df, y_train_series = self._dataemitter.emit_train_Xy(
                dropfirst=self._dropfirst, verbose=verbose
            )
            y_train = y_train_series.to_numpy()

            if self._feature_selectors is not None:
                self._feature_selection_report = VotingSelectionReport(
                    selectors=self._feature_selectors,
                    dataemitter=self._dataemitter,
                    max_n_features=self._max_n_features,
                    verbose=verbose,
                )
                X_train = self._feature_selection_report._emit_train_X(
                    dropfirst=self._dropfirst, verbose=verbose
                )
                self._predictors = self._feature_selection_report.top_features()
            else:
                self._predictors = X_train_df.columns.to_list()
                X_train = X_train_df

            y_train_encoded = self._label_encoder.fit_transform(y_train)

            self._hyperparam_searcher.fit(X_train, y_train_encoded, verbose)
            self._best_estimator = self._hyperparam_searcher._best_estimator

            y_pred_encoded = self._best_estimator.predict(X_train)
            if hasattr(self._best_estimator, "predict_proba"):
                y_pred_score = self._best_estimator.predict_proba(X_train)

                if self._is_binary:
                    self._threshold = select_optimal_threshold_binary(
                        y_train_encoded,
                        y_pred_score[:, 1],
                        metric=self._threshold_strategy,
                    )
                    y_pred_encoded = predict_with_threshold_binary(
                        y_pred_score[:, 1], self._threshold
                    )

                    if verbose and self._threshold_strategy is not None:
                        print_wrapped(
                            f"Optimal threshold set for {self._name} "
                            f"via {self._threshold_strategy}.",
                            type="PROGRESS",
                        )

                else:
                    self._threshold = select_optimal_thresholds_multiclass(
                        y_train_encoded, y_pred_score, metric=self._threshold_strategy
                    )
                    y_pred_encoded = predict_with_thresholds_multiclass(
                        y_pred_score, thresholds=self._threshold
                    )

                    if verbose and self._threshold_strategy is not None:
                        print_wrapped(
                            f"Optimal thresholds set for {self._name} via "
                            f"{self._threshold_strategy}.",
                            type="PROGRESS",
                        )

            elif hasattr(self._best_estimator, "decision_function"):
                y_pred_score = self._best_estimator.decision_function(X_train)

            if self._is_binary:
                self._train_scorer = ClassificationBinaryScorer(
                    y_pred=y_pred_encoded,
                    y_true=y_train_encoded,
                    pos_label=self._label_encoder.classes_[1],
                    y_pred_score=y_pred_score,
                    name=str(self),
                )
            else:
                self._train_scorer = ClassificationMulticlassScorer(
                    y_pred=self._label_encoder.inverse_transform(y_pred_encoded),
                    y_true=y_train,
                    y_pred_score=y_pred_score,
                    y_pred_class_order=self._label_encoder.classes_,
                    name=str(self),
                )

        else:
            raise ValueError("The datahandler must not be None")

        X_test_df, y_test_series = self._dataemitter.emit_test_Xy(
            dropfirst=self._dropfirst, verbose=verbose
        )

        y_test = y_test_series.to_numpy()

        if self._feature_selectors is None:
            X_test = X_test_df
        else:
            X_test = self._feature_selection_report._emit_test_X(
                dropfirst=self._dropfirst, verbose=verbose
            )

        y_pred_score = None
        if hasattr(self._best_estimator, "predict_proba"):
            y_pred_score = self._best_estimator.predict_proba(X_test)

            if self._is_binary:
                y_pred_encoded = predict_with_threshold_binary(
                    y_pred_score[:, 1], threshold=self._threshold
                )
            else:
                y_pred_encoded = predict_with_thresholds_multiclass(
                    y_pred_score, thresholds=self._threshold
                )

        elif hasattr(self._best_estimator, "decision_function"):
            y_pred_score = self._best_estimator.decision_function(X_test)
            y_pred_encoded = self._best_estimator.predict(X_test)

        if self._is_binary:
            self._test_scorer = ClassificationBinaryScorer(
                y_pred=y_pred_encoded,
                y_true=self._label_encoder.transform(y_test),
                pos_label=self._label_encoder.classes_[1],
                y_pred_score=y_pred_score,
                name=str(self),
            )
        else:
            self._test_scorer = ClassificationMulticlassScorer(
                y_pred=self._label_encoder.inverse_transform(y_pred_encoded),
                y_true=y_test,
                y_pred_score=y_pred_score,
                y_pred_class_order=self._label_encoder.classes_,
                name=str(self),
            )

    def sklearn_estimator(self):
        """Returns the best estimator (sklearn estimator object). The best estimator
        was fitted on the train data through the hyperparameter search process.

        Note that the sklearn estimator can be saved and used for future predictions.
        However, the input data must be preprocessed in the same way. If you intend
        to use the estimator for future predictions, it is recommended that you
        manually specify every preprocessing step, which will ensure that you
        have full control over how the data is being transformed for future
        reproducibility and predictions.

        It is not recommended to use TableMage for ML production.
        We recommend using TableMage to quickly identify promising models
        and then manually implementing and training
        the best model in a production environment.

        Returns
        -------
        BaseEstimator
        """
        return self._best_estimator

    def sklearn_pipeline(self) -> Pipeline:
        """Returns an sklearn pipeline object. The pipeline allows for
        retrieving model predictions directly from data formatted like the original
        train and test data.

        The pipeline is composed of the following steps:
            1. Custom data preprocessing steps.
            2. Hyperparameter search object.
            3. The best model determined from the hyperparameter search process.


        It is not recommended to use TableMage for ML production.
        We recommend using TableMage to quickly identify promising models
        and then manually implementing and training
        the best model in a production environment.

        Returns
        -------
        Pipeline
        """
        if self._feature_selectors is not None:
            pipeline = Pipeline(
                steps=[
                    (
                        "custom_prep_data",
                        self._dataemitter.sklearn_preprocessing_transformer(),
                    ),
                    (
                        "feature_selector",
                        ColumnSelector(self._feature_selection_report.top_features()),
                    ),
                    ("model", self._hyperparam_searcher._searcher),
                ]
            )
        else:
            pipeline = Pipeline(
                steps=[
                    (
                        "custom_prep_data",
                        self._dataemitter.sklearn_preprocessing_transformer(),
                    ),
                    ("model", self._hyperparam_searcher._searcher),
                ]
            )
        return pipeline

    def hyperparam_searcher(self) -> HyperparameterSearcher:
        """Returns the HyperparameterSearcher object.

        Returns
        -------
        HyperparameterSearcher
        """
        return self._hyperparam_searcher

    def _set_voting_selection_report(
        self, voting_selection_report: VotingSelectionReport
    ):
        """Adds a VotingSelectionReport object to the model. The VotingSelectionReport
        must have already been fitted to the data.

        Parameters
        ----------
        voting_selection_report : VotingSelectionReport
            The VotingSelectionReport object that has already been fitted to the data.
        """
        self._feature_selection_report = voting_selection_report

    def fs_report(self) -> VotingSelectionReport | None:
        """Returns the VotingSelectionReport object.

        Returns
        -------
        VotingSelectionReport | None
            None if the VotingSelectionReport object has not been set (e.g. no
            feature selection was conducted).
        """
        if self._feature_selection_report is None:
            print_wrapped(
                f"No feature selection report available for {self._name}.",
                type="WARNING",
            )
        return self._feature_selection_report

    def is_binary(self) -> bool:
        """Returns True if the model is binary.

        Returns
        -------
        bool
            True if the model is binary.
        """
        return self._is_binary

    def is_cross_validated(self) -> bool:
        """Returns True if the cross validation metrics are available.

        Returns
        -------
        bool
            True if cross validation metrics are available.
        """
        return self._dataemitters is not None

    def pos_label(self) -> str | None:
        """Returns the positive label. This method is only for binary
        classification models. A warning is printed if the model is not binary, and
        None is returned.

        Returns
        -------
        str | None
            The positive label. None if the model is not binary.
        """
        if self._is_binary:
            return self._label_encoder.classes_[1]
        else:
            print_wrapped(
                "This is not a binary model. No positive label available.",
                type="WARNING",
            )
            return None

    def predictors(self) -> list[str] | None:
        """Returns a list predictor variable names. A warning is printed if the model
        has not been fitted, and None is returned.

        Returns
        -------
        list[str] | None
            A list of predictor variable names used in the final model,
            after feature selection and data transformation.
        """
        if self._predictors is None:
            print_wrapped(
                "No predictors available. The model has not been fitted.",
                type="WARNING",
            )
        return self._predictors

    def feature_importance(self) -> pd.DataFrame | None:
        """Returns the feature importances of the best estimator.
        If the best estimator is a linear model, the coefficients are
        returned.

        Returns
        -------
        pd.DataFrame | None
            A DataFrame with feature importances or coefficients. None if no importances
            or coefficients are available.
        """
        if self._best_estimator is None:
            raise RuntimeError("Model has not been fitted.")

        if hasattr(self._best_estimator, "feature_importances_"):
            importances = self._best_estimator.feature_importances_
            type = "Importances"
            self._feature_importance_type_str = "feature importances"
        elif hasattr(self._best_estimator, "coef_"):
            importances = self._best_estimator.coef_
            type = "Coefficients"
            self._feature_importance_type_str = "coefficients"
        else:
            print_wrapped(
                "No feature importances or coefficients are available for "
                f"{color_text(self._name, 'yellow')}.",
            )
            return None

        importances_shape = importances.shape
        # handle case of multiple classes
        if len(importances_shape) == 2:
            if importances_shape[0] == 1:
                importances = importances[0]
            else:
                importances = importances.mean(axis=0)

        return pd.DataFrame(
            data=importances,
            index=pd.Series(self._predictors, name="Feature"),
            columns=[type],
        )

    def _to_dict(self):
        """Returns a dictionary representation of the model.

        Returns
        -------
        dict
            A dictionary representation of the model.
        """
        output = {
            "name": self._name,
            "predictors": self._predictors,
        }

        if self.feature_importance() is not None:
            output[self._feature_importance_type_str] = (
                self.feature_importance().to_dict()
            )

        if self._hyperparam_searcher is not None:
            output["fitting_details"] = self._hyperparam_searcher._to_dict()

        # for all floats in dict, round to print_options._n_decimals
        def round_floats(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    round_floats(value)
                elif isinstance(value, float):
                    d[key] = np.round(value, print_options._n_decimals)

        round_floats(output)
        return output

    def _validate_inputs(self):
        """Ensures that the inputs are valid.

        Raises
        ------
        ValueError
            If the inputs are invalid.
        """
        if self._feature_selectors is not None:
            if not isinstance(self._feature_selectors, list):
                raise ValueError("feature_selectors must be a list.")
            for selector in self._feature_selectors:
                if not isinstance(selector, BaseFSC):
                    raise ValueError(
                        "Each feature selector must be a BaseFSC object. "
                        f"Got {type(selector)}."
                    )

    def __str__(self):
        return self._name
