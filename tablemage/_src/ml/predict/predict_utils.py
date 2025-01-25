from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Selects columns from a DataFrame. Used for pipeline compatibility."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]


class InverseTransformRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model: Pipeline | None = None, inverse_func=None):
        self.model = model
        self.inverse_func = inverse_func

    def predict(self, X):
        y_pred_transformed = self.model.predict(X)
        return (
            self.inverse_func(y_pred_transformed)
            if self.inverse_func
            else y_pred_transformed
        )

    def score(self, X, y):
        return self.model.score(X, y)
