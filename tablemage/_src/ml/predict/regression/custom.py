from sklearn.model_selection._search import BaseSearchCV
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from .base import BaseR
from ....metrics import RegressionScorer
from ....display.print_utils import print_wrapped, quote_and_color
from ..predict_utils import InverseTransformRegressor


class CustomR(BaseR):
    """TableMage-compatible interface for user-designed scikit-learn
    estimators/searches/pipelines for regression.

    Hyperparameter search is not conducted unless provided by the
    estimator.
    """

    def __init__(
        self,
        estimator: BaseEstimator | BaseSearchCV | Pipeline,
        name: str | None = None,
    ):
        """Initializes a CustomR object.

        Parameters
        ----------
        estimator : BaseEstimator | BaseSearchCV | Pipeline
            The estimator to be used. Must have a fit method and a
            predict method.

        name : str
            Default: None.
            The name of the model. If None, the estimator's
            __str__() implementation is used.
        """
        super().__init__()
        self._best_estimator = estimator
        if name is None:
            self._name = str(estimator)
        else:
            self._name = name

    def fit(self, verbose: bool = False):
        """Fits the model. Records training metrics, which can be done via
        nested cross validation.

        Parameters
        ----------
        verbose : bool.
            Default: False. If True, prints progress.
        """
        y_scaler = self._dataemitter.y_scaler()
        dropfirst = False

        if self._dataemitters is None and self._dataemitter is not None:
            X_train_df, y_train_series = self._dataemitter.emit_train_Xy(
                dropfirst=dropfirst,
                verbose=verbose,
            )
            self._predictors = X_train_df.columns.to_list()
            X_train = X_train_df
            y_train = y_train_series.to_numpy()
            if verbose:
                print_wrapped(
                    f"Fitting {quote_and_color(self._name)}.", type="PROGRESS"
                )
            self._best_estimator.fit(X_train, y_train)
            y_pred = self._best_estimator.predict(X_train)
            if y_scaler is not None:
                y_pred = y_scaler.inverse_transform(y_pred)
                y_train = y_scaler.inverse_transform(y_train)
            self._train_scorer = RegressionScorer(
                y_pred=y_pred,
                y_true=y_train,
                n_predictors=X_train.shape[1],
                name=str(self),
            )

        elif self._dataemitters is not None and self._dataemitter is not None:
            y_preds = []
            y_trues = []
            for emitter in self._dataemitters:
                (
                    X_train_df,
                    y_train_series,
                    X_test_df,
                    y_test_series,
                ) = emitter.emit_train_test_Xy(
                    dropfirst=dropfirst,
                    verbose=verbose,
                )
                X_train = X_train_df
                y_train = y_train_series.to_numpy()
                X_test = X_test_df
                y_test = y_test_series.to_numpy()
                if verbose:
                    print_wrapped(
                        f"Fitting {quote_and_color(self._name)}.", type="PROGRESS"
                    )
                self._best_estimator.fit(X_train, y_train)
                y_pred = self._best_estimator.predict(X_test)
                if y_scaler is not None:
                    y_pred = y_scaler.inverse_transform(y_pred)
                    y_test = y_scaler.inverse_transform(y_test)

                y_preds.append(y_pred)
                y_trues.append(y_test)

            self._cv_scorer = RegressionScorer(
                y_pred=y_preds,
                y_true=y_trues,
                n_predictors=X_train.shape[1],
                name=str(self),
            )

            # refit on all data
            X_train_df, y_train_series = self._dataemitter.emit_train_Xy(
                dropfirst=dropfirst,
                verbose=verbose,
            )
            self._predictors = X_train_df.columns.to_list()
            X_train = X_train_df
            y_train = y_train_series.to_numpy()
            if verbose:
                print_wrapped(
                    f"Fitting {quote_and_color(self._name)}.", type="PROGRESS"
                )
            self._best_estimator.fit(X_train, y_train)
            y_pred = self._best_estimator.predict(X_train)
            if y_scaler is not None:
                y_pred = y_scaler.inverse_transform(y_pred)
                y_train = y_scaler.inverse_transform(y_train)

            self._train_scorer = RegressionScorer(
                y_pred=y_pred,
                y_true=y_train,
                n_predictors=X_train.shape[1],
                name=str(self),
            )

        else:
            raise ValueError("The datahandler must not be None")

        X_test_df, y_test_series = self._dataemitter.emit_test_Xy(
            dropfirst=dropfirst,
            verbose=verbose,
        )
        X_test = X_test_df
        y_test = y_test_series.to_numpy()

        y_pred = self._best_estimator.predict(X_test)
        if y_scaler is not None:
            y_pred = y_scaler.inverse_transform(y_pred)
            y_test = y_scaler.inverse_transform(y_test)

        self._test_scorer = RegressionScorer(
            y_pred=y_pred, y_true=y_test, n_predictors=X_test.shape[1], name=str(self)
        )

    def sklearn_pipeline(self) -> Pipeline:
        """Returns an sklearn pipeline object. The pipelien allows for
        retrieving model predictions directly from data formatted like the original
        train and test data.

        It is not recommended to use TableMage for ML production.
        We recommend using TableMage to quickly identify promising models
        and then manually implementing and training
        the best model in a production environment.

        Returns
        -------
        Pipeline
        """
        if isinstance(self._best_estimator, Pipeline):
            new_step = (
                "custom_prep_data",
                self._dataemitter.sklearn_preprocessing_transformer(),
            )
            pipeline = Pipeline(steps=[new_step, ("model", self._best_estimator)])
        else:
            pipeline = Pipeline(
                steps=[
                    (
                        "custom_prep_data",
                        self._dataemitter.sklearn_preprocessing_transformer(),
                    ),
                    ("model", self._best_estimator),
                ]
            )
        if self._dataemitter.y_scaler() is not None:
            pipeline = InverseTransformRegressor(
                model=pipeline,
                inverse_func=self._dataemitter.y_scaler().inverse_transform,
            )
        return pipeline

    def hyperparam_searcher(self):
        """Raises NotImplementedError. Not implemented for CustomR."""
        raise NotImplementedError("CustomR has no HyperparameterSearcher.")
