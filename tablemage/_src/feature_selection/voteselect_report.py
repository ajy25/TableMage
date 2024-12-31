import pandas as pd
from . import BaseFS
from ..data.datahandler import DataEmitter
from ..display.print_utils import (
    print_wrapped,
    quote_and_color,
    color_text,
    bold_text,
    fill_ignore_format,
    list_to_string,
)
from ..display.print_options import print_options


class VotingSelectionReport:
    """Class for generating feature selection-relevant tables."""

    def __init__(
        self,
        selectors: list[BaseFS],
        dataemitter: DataEmitter,
        max_n_features: int | None = None,
        verbose: bool = True,
    ):
        """Initializes a VotingSelectionReport object.
        VotingSelectionReport selects features via voting selection.

        Parameters
        ----------
        selectors : list[BaseSelector]
            Each BaseSelector decides on a maximum of max_n_features.

        dataemitter : DataEmitter
            The DataEmitter object that contains the data.

        max_n_features : int | None
            Default: None.
            Number of desired features. 0 < max_n_features < n_predictors.
            If None, then all features with at least 50% support are selected.

        verbose : bool
            Default: True. If True, prints progress.
        """
        self._selector_to_support = {}
        self._emitter = dataemitter

        self._y_var = self._emitter.y_var()
        self._predictors = self._emitter.X_vars()
        X_train_df = self._emitter.emit_train_X(verbose=False)
        if len(X_train_df) == 0:
            raise ValueError(
                "No data was emitted. All rows with missing values were dropped. "
                + "This may have resulted in an empty dataset. "
                + "Please consider removing highly missing variables "
                + "or imputing missing values."
            )

        self._selectors = selectors

        for selector in selectors:
            if verbose:
                print_wrapped(f"Fitting {quote_and_color(selector)}.", type="PROGRESS")
            features, _, support = selector.select(self._emitter)
            self._selector_to_support[str(selector)] = support
        self._all_features = features

        self._votes_df = pd.DataFrame.from_dict(
            self._selector_to_support, orient="index", columns=features
        )
        self._vote_counts_series = self._votes_df.sum(axis=0)

        self._selector_dict_indexable_by_str = {
            str(selector): selector for selector in selectors
        }
        if max_n_features is not None:
            self._top_features = self._vote_counts_series.sort_values(
                ascending=False
            ).index.to_list()[:max_n_features]
        else:
            self._top_features = self._vote_counts_series[
                self._vote_counts_series >= len(selectors) / 2
            ].index.to_list()

    def top_features(self) -> list:
        """Returns a list of top features determined by the voting
        selectors.

        Returns
        -------
        list
            Top features.
        """
        return self._top_features

    def all_features(self) -> list:
        """Returns a list of all features considered by the voting
        selectors.

        Returns
        -------
        list
            All features.
        """
        return self._all_features

    def votes(self) -> pd.DataFrame:
        """Returns a DataFrame that describes the distribution of
        votes among selectors.

        Returns
        -------
        pd.DataFrame
            Votes DataFrame.
        """
        return self._votes_df.T

    def _emit_train_X(
        self, dropfirst: bool = True, verbose: bool = True
    ) -> pd.DataFrame:
        """Returns the training DataFrame with only the top features."""
        return self._emitter.emit_train_Xy(dropfirst=dropfirst, verbose=verbose)[0][
            self._top_features
        ]

    def _emit_test_X(
        self, dropfirst: bool = True, verbose: bool = True
    ) -> pd.DataFrame:
        """Returns the test DataFrame with only the top features."""
        return self._emitter.emit_test_Xy(dropfirst=dropfirst, verbose=verbose)[0][
            self._top_features
        ]

    def __getitem__(self, index: str) -> BaseFS:
        """Returns the RegressionBaseSelector by nickname index."""
        return self._selector_dict_indexable_by_str[index]

    def __str__(self) -> str:
        n_dec = print_options._n_decimals
        max_width = print_options._max_line_width

        top_divider = color_text("=" * max_width, "none") + "\n"
        bottom_divider = "\n" + color_text("=" * max_width, "none")
        divider = "\n" + color_text("-" * max_width, "none") + "\n"
        divider_invisible = "\n" + " " * max_width + "\n"

        title_message = bold_text("Voting Selection Report")

        target_var = "'" + self._y_var + "'"
        target_message = f"{bold_text('Target variable:')}\n"
        target_message += fill_ignore_format(
            color_text(target_var, "purple"),
            width=max_width,
            initial_indent=2,
            subsequent_indent=2,
        )

        predictors_message = f"{bold_text('Candidate predictor variables:')}\n"
        predictors_message += fill_ignore_format(
            list_to_string(self._predictors),
            width=max_width,
            initial_indent=2,
            subsequent_indent=2,
        )

        models_str = list_to_string(
            [model._name for model in self._selectors],
            color="blue",
        )
        models_message = f"{bold_text('Feature selectors:')}\n"
        models_message += fill_ignore_format(
            models_str,
            width=max_width,
            initial_indent=2,
            subsequent_indent=2,
        )

        selected_features_message = f"{bold_text('Selected features:')}\n"
        selected_features_message += fill_ignore_format(
            list_to_string(self._top_features, color="purple"),
            width=max_width,
            initial_indent=2,
            subsequent_indent=2,
        )

        final_message = (
            top_divider
            + title_message
            + divider
            + target_message
            + divider_invisible
            + predictors_message
            + divider_invisible
            + models_message
            + divider
            + selected_features_message
            + bottom_divider
        )

        return final_message

    def _to_dict(self) -> dict:
        """Returns the object as a dictionary."""
        return {
            "target": self._y_var,
            "candidate_predictors": self._predictors,
            "feature_selectors": [str(selector) for selector in self._selectors],
            "selected_features": self._top_features,
        }

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))
