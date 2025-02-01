import pandas as pd
import numpy as np
from typing import Literal
from copy import deepcopy
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils._testing import ignore_warnings
from ..display.print_utils import (
    print_wrapped,
    color_text,
    quote_and_color,
    bold_text,
    list_to_string,
    fill_ignore_format,
)
from .preprocessing import (
    BaseSingleVarScaler,
    Log1PTransformSingleVar,
    LogTransformSingleVar,
    MinMaxSingleVar,
    StandardizeSingleVar,
    CustomOneHotEncoder,
    RobustStandardizeSingleVar,
    NormalQuantileTransformSingleVar,
    UniformQuantileTransformSingleVar,
    CombinedSingleVarScaler,
)
from ..display.print_options import print_options
from .dataemitter import DataEmitter, PreprocessStepTracer
from .utils.formula import parse_formula, color_and_quote_formula_vars
from .utils.var_naming import rename_vars, rename_var


class DataHandler:
    """DataHandler: a class that handles all aspects of data
    preprocessing and loading."""

    def __init__(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        name: str | None = None,
        verbose: bool = True,
    ):
        """Initializes a DataHandler object.

        Parameters
        ----------
        df_train : pd.DataFrame
            The train DataFrame.

        df_test : pd.DataFrame
            The test DataFrame.

        name : str | None
            Default: None. The name of the DataHandler object.

        verbose : bool
            Default: True. If True, prints updates and warnings.
        """
        # force align columns
        df_train = df_train.copy()
        df_test = df_test[df_train.columns].copy()

        self._checkpoint_name_to_df: dict[
            str, tuple[pd.DataFrame, pd.DataFrame]
        ] = dict()
        self._verbose = verbose

        # force bool to int, object to category
        bool_cols = df_train.select_dtypes(include=["bool"]).columns
        df_train[bool_cols] = df_train[bool_cols].astype(int)
        df_test[bool_cols] = df_test[bool_cols].astype(int)

        # verify and set the original DataFrames
        self._verify_input_dfs(df_train, df_test)

        self._orig_df_train, self._orig_df_test = self._rename_varnames(
            df_train, df_test
        )

        self._orig_df_train, self._orig_df_test = self._force_train_test_var_agreement(
            self._orig_df_train, self._orig_df_test
        )

        (
            self._categorical_vars,
            self._numeric_vars,
            self._categorical_to_categories,
        ) = self._compute_categorical_numeric_vars(self._orig_df_train)

        # set the working DataFrames
        self._working_df_train = self._orig_df_train.copy()
        self._working_df_test = self._orig_df_test.copy()

        # keep track of scalers
        self._numeric_var_to_scalers = {var: [] for var in self._numeric_vars}

        # set the name
        if name is None:
            self._name = "Unnamed Dataset"
        else:
            self._name = name

        # step tracing for all preprocessing steps
        self._preprocess_step_tracer = PreprocessStepTracer()
        self._preprocess_step_tracer.add_category_mapping(
            self._categorical_to_categories
        )

    # --------------------------------------------------------------------------
    # CHECKPOINT HANDLING
    # --------------------------------------------------------------------------
    def load_data_checkpoint(self, checkpoint: str | None = None) -> "DataHandler":
        """The working train and working test DataFrames are reset to the
        original input DataFrames given at object initialization.

        Parameters
        ----------
        checkpoint : str | None
            Default: None. If None, sets the working DataFrames to the original
            DataFrames given at object initialization.

        Returns
        -------
        DataHandler
            Returns self for method chaining.
        """
        if checkpoint is None:
            self._working_df_test = self._orig_df_test.copy()
            self._working_df_train = self._orig_df_train.copy()
            self._numeric_var_to_scalers = {var: [] for var in self._numeric_vars}
            self._preprocess_step_tracer = PreprocessStepTracer()
            if self._verbose:
                shapes_dict = self._shapes_str_formatted()
                print_wrapped(
                    "Datasets reset to original state. "
                    + "Train, test shapes: "
                    + f'{shapes_dict["train"]}, {shapes_dict["test"]}.',
                    type="UPDATE",
                )
        else:
            self._working_df_test = self._checkpoint_name_to_df[checkpoint][0].copy()
            self._working_df_train = self._checkpoint_name_to_df[checkpoint][1].copy()
            self._numeric_var_to_scalers = self._checkpoint_name_to_df[checkpoint][
                2
            ].copy()
            self._preprocess_step_tracer: PreprocessStepTracer = (
                self._checkpoint_name_to_df[checkpoint][3].copy()
            )
            if self._verbose:
                shapes_dict = self._shapes_str_formatted()
                print_wrapped(
                    "Datasets reset to state at checkpoint "
                    + f'{quote_and_color(checkpoint, "yellow")}. '
                    + "Train, test shapes: "
                    + f'{shapes_dict["train"]}, {shapes_dict["test"]}.',
                    type="UPDATE",
                )
        (
            self._categorical_vars,
            self._numeric_vars,
            self._categorical_to_categories,
        ) = self._compute_categorical_numeric_vars(self._working_df_train)

        return self

    def save_data_checkpoint(self, checkpoint: str) -> "DataHandler":
        """Saves the current state of the working train and test DataFrames.
        The state may be returned to by calling
        load_data_checkpoint(checkpoint).

        Parameters
        ----------
        checkpoint : str
            Name of the checkpoint.

        Returns
        -------
        DataHandler
            Returns self for method chaining.
        """
        if self._verbose:
            print_wrapped(
                "Saved data checkpoint " + f'{quote_and_color(checkpoint, "yellow")}.',
                type="UPDATE",
            )
        self._checkpoint_name_to_df[checkpoint] = (
            self._working_df_test.copy(),
            self._working_df_train.copy(),
            self._numeric_var_to_scalers.copy(),
            self._preprocess_step_tracer.copy(),
        )

        return self

    def remove_data_checkpoint(self, checkpoint: str) -> "DataHandler":
        """Removes a saved checkpoint to conserve memory.

        Parameters
        ----------
        checkpoint : str
            Name of the checkpoint to remove.

        Returns
        -------
        DataHandler
            Returns self for method chaining.
        """
        self._checkpoint_name_to_df.pop(checkpoint)
        if self._verbose:
            print_wrapped(
                "Removed working DataFrames checkpoint "
                + f'{quote_and_color(checkpoint, "yellow")}.',
                type="UPDATE",
            )
        return self

    # --------------------------------------------------------------------------
    # GETTERS
    # --------------------------------------------------------------------------

    def df_all(self) -> pd.DataFrame:
        """Returns the working train and test DataFrames concatenated.

        Returns
        -------
        pd.DataFrame
            Concatenated DataFrames.
        """
        no_test = True
        for a, b in zip(self._working_df_train.index, self._working_df_test.index):
            if a != b:
                no_test = False
                break
        if no_test:
            out = self.df_train()
        else:
            out = pd.concat([self.df_train(), self.df_test()])
        return out

    def df_train(self) -> pd.DataFrame:
        """Returns the working train DataFrame.

        Returns
        -------
        pd.DataFrame
            The working train DataFrame.
        """
        return self._working_df_train

    def df_test(self) -> pd.DataFrame:
        """Returns the working test DataFrame.

        Returns
        -------
        pd.DataFrame
            The working test DataFrame.
        """
        return self._working_df_test

    def vars(self) -> list[str]:
        """Returns a list of all variables in the working DataFrames.

        Returns
        -------
        list[str]
            List of variable names.
        """
        out = self._working_df_train.columns.to_list()
        return out

    def numeric_vars(self) -> list[str]:
        """Returns copy of list of numeric variables.

        Returns
        -------
        list[str]
            List of numeric variable names.
        """
        out = self._numeric_vars.copy()
        return out

    def categorical_vars(self) -> list[str]:
        """Returns copy of list of categorical variables.

        Returns
        -------
        list[str]
            List of categorical variable names.
        """
        out = self._categorical_vars.copy()
        return out

    def scaler(self, var: str) -> CombinedSingleVarScaler | None:
        """Returns the scaler for a numeric variable, which could be None.

        Parameters
        ----------
        var : str
            Name of the variable.

        Returns
        -------
        CombinedSingleVarScaler | None
        """
        if len(self._numeric_var_to_scalers[var]) == 0:
            return None
        return CombinedSingleVarScaler(self._numeric_var_to_scalers[var])

    def train_test_emitter(self, y_var: str | None, X_vars: list[str]) -> DataEmitter:
        """Returns a DataEmitter object for the working train DataFrame and
        the working test DataFrame.

        Parameters
        ----------
        y_var : str | None
            Name of the target variable.

        X_vars : list[str]
            Names of the predictor variables.

        Returns
        -------
        DataEmitter
        """
        if y_var is not None:
            if y_var not in self._working_df_train.columns:
                raise ValueError(f"Invalid target variable name: {y_var}.")
        for var in X_vars:
            if var not in self._working_df_train.columns:
                raise ValueError(f"Invalid variable name: {var}.")
        return DataEmitter(
            self._orig_df_train,
            self._orig_df_test,
            y_var,
            X_vars,
            self._preprocess_step_tracer,
        )

    def full_dataset_emitter(self, y_var: str | None, X_vars: list[str]) -> DataEmitter:
        """Returns a DataEmitter object for the working train DataFrame and
        the working test DataFrame.

        The concatenated DataFrame (full DataFrame) is re-preprocessed as if
        it were the original train DataFrame. Note that the "test" DataFrame is the
        same as the "train" DataFrame for the outputted DataEmitter in this case.

        Parameters
        ----------
        y_var : str | None
            Name of the target variable.

        X_vars : list[str]
            Names of the predictor variables.

        Returns
        -------
        DataEmitter
        """
        if y_var is not None:
            if y_var not in self._working_df_train.columns:
                raise ValueError(f"Invalid target variable name: {y_var}.")
        for var in X_vars:
            if var not in self._working_df_train.columns:
                raise ValueError(f"Invalid variable name: {var}.")
        if self._orig_df_train.index.equals(self._orig_df_test.index):
            if self._verbose:
                print_wrapped(
                    text=(
                        "Train and test DataFrames have the same index. "
                        "Assuming Analyzer was initialized with a single DataFrame. "
                        "Returning a DataEmitter object with only the train DataFrame."
                    ),
                    type="WARNING",
                )
            concatinated_df = self._orig_df_train
        else:
            concatinated_df = pd.concat([self._orig_df_train, self._orig_df_test])
        return DataEmitter(
            concatinated_df,
            concatinated_df,
            y_var,
            X_vars,
            self._preprocess_step_tracer,
        )

    def kfold_emitters(
        self,
        y_var: str,
        X_vars: list[str],
        n_folds: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> list[DataEmitter]:
        """Returns a list of DataEmitter objects for cross-validation.
        DataEmitter objects are built from KFold
        (StratifiedKFold if target is categorical) applied to
        the working train DataFrame.

        Parameters
        ----------
        y_var : str
            Name of the target variable.

        X_vars : list[str]
            Names of the predictor variables.

        n_folds : int
            Default: 5. Number of folds.

        shuffle : bool
            Default: True. Whether to shuffle the data.

        random_state : int
            Default: 42. Random state for the
            KFold/StratifiedKFold. Ignored if shuffle is False.

        Returns
        -------
        list[DataEmitter]
        """
        if n_folds < 2:
            raise ValueError("n_folds must be at least 2.")

        if y_var not in self._working_df_train.columns:
            raise ValueError(f"Invalid target variable name: {y_var}.")
        for var in X_vars:
            if var not in self._working_df_train.columns:
                raise ValueError(f"Invalid variable name: {var}.")

        use_stratified = False
        if y_var in self._orig_df_train.columns:
            if self._orig_df_train[y_var].dtype in ["object", "category", "bool"]:
                use_stratified = True
        if use_stratified:
            try:
                if shuffle:
                    kf = StratifiedKFold(
                        n_splits=n_folds, random_state=random_state, shuffle=True
                    )
                else:
                    kf = StratifiedKFold(n_splits=n_folds, shuffle=False)

                out = []
                for train_index, test_index in kf.split(
                    self._orig_df_train, self._orig_df_train[y_var]
                ):
                    df_train = self._orig_df_train.iloc[train_index]
                    df_test = self._orig_df_train.iloc[test_index]
                    out.append(
                        DataEmitter(
                            df_train,
                            df_test,
                            y_var,
                            X_vars,
                            self._preprocess_step_tracer,
                        )
                    )
                return out

            except ValueError as e:
                if self._verbose:
                    print_wrapped(
                        f"StratifiedKFold failed: {e}. Using KFold instead.",
                        type="WARNING",
                    )
                use_stratified = False
                if shuffle:
                    kf = KFold(
                        n_splits=n_folds, random_state=random_state, shuffle=True
                    )
                else:
                    kf = KFold(n_splits=n_folds, shuffle=False)

                out = []
                for train_index, test_index in kf.split(self._orig_df_train):
                    df_train = self._orig_df_train.iloc[train_index]
                    df_test = self._orig_df_train.iloc[test_index]
                    out.append(
                        DataEmitter(
                            df_train,
                            df_test,
                            y_var,
                            X_vars,
                            self._preprocess_step_tracer,
                        )
                    )
                return out

        else:
            if shuffle:
                kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=True)
            else:
                kf = KFold(n_splits=n_folds, shuffle=False)

            out = []
            for train_index, test_index in kf.split(self._orig_df_train):
                df_train = self._orig_df_train.iloc[train_index]
                df_test = self._orig_df_train.iloc[test_index]
                out.append(
                    DataEmitter(
                        df_train, df_test, y_var, X_vars, self._preprocess_step_tracer
                    )
                )
            return out

    def is_binary(self, var: str, is_numeric: bool = False) -> bool:
        """Checks if a given variable is binary (i.e., it only has two unique values).

        Parameters
        ----------
        var : str
            Name of the variable.

        is_numeric : bool
            Whether the variable is numeric binary (i.e., has only 0 and 1 values). \
            Default: False.

        Returns
        -------
        bool
            True if the input variable is binary.
        """
        if is_numeric:
            unique_vals = self.df_all()[var].unique()
            return len(unique_vals) == 2 and 0 in unique_vals and 1 in unique_vals
        return len(self.df_all()[var].unique()) == 2

    # --------------------------------------------------------------------------
    # PREPROCESSING and FEATURE ENGINEERING
    # --------------------------------------------------------------------------

    def engineer_numeric_feature(
        self,
        feature_name: str,
        formula: str,
    ) -> "DataHandler":
        """Engineers a new feature based on a formula. The formula
        can only involve numeric variables. Yields a new numeric variable.

        Parameters
        ----------
        feature_name : str
            The name of the new variable engineered.

        formula : str
            Formula for the new feature. For example, "x1 + x2" would create
            a new feature that is the sum of the columns x1 and x2 in the DataFrame.
            All variables used must be numeric.
            Handles the following operations:

            - Arithmetic expressions, yielding a new numeric variable
                - Addition (+)
                - Subtraction (-)
                - Multiplication (*)
                - Division (/)
                - Parentheses ()
                - Exponentiation (**)
                - Logarithm (log)
                - Exponential (exp)
                - Square root (sqrt)

            If the i-th unit is missing a value in any of the variables used in the
            formula, then the i-th unit of the new feature will be missing.

        Examples
        --------
        >>> analyzer.engineer_feature("x3", "x1 + x2")
        >>> assert "x3" in analyzer.datahandler.vars()
        True
        >>> assert analyzer.datahandler.df_train()["x3"].equals(
        ...     analyzer.datahandler.df_train()["x1"] + analyzer.datahandler.df_train()["x2"]
        ... )
        True

        Returns
        -------
        DataHandler
            Returns self for method chaining.
        """
        feature_name = rename_var(feature_name)

        if feature_name in self._working_df_train.columns:
            print_wrapped(
                f"Variable {quote_and_color(feature_name, 'purple')} already exists. "
                "Overwriting.",
                type="WARNING",
                level="INFO",
            )

        self._working_df_train[feature_name] = parse_formula(
            formula, self._working_df_train
        )
        self._working_df_test[feature_name] = parse_formula(
            formula, self._working_df_test
        )

        if self._verbose:
            print_wrapped(
                f"Engineered numeric variable "
                + quote_and_color(feature_name, "purple")
                + color_text(" = ", "yellow")
                + color_and_quote_formula_vars(formula)
                + ".",
                type="UPDATE",
            )

        (
            self._categorical_vars,
            self._numeric_vars,
            self._categorical_to_categories,
        ) = self._compute_categorical_numeric_vars(self._working_df_train)

        self._preprocess_step_tracer.add_step(
            "engineer_numeric_feature",
            {
                "feature_name": feature_name,
                "formula": formula,
            },
        )

        return self

    def engineer_categorical_feature(
        self,
        feature_name: str,
        numeric_var: str,
        level_names: list[str],
        thresholds: list[float],
        leq: bool = False,
    ) -> "DataHandler":
        """Engineers a new categorical feature based on a list of thresholds
        for a single numeric variable. The thresholds must be in ascending order.

        Parameters
        ----------
        feature_name : str
            The name of the new variable engineered.

        numeric_var : str
            The name of the numeric variable.

        level_names : list[str]
            The names of the levels of the new categorical variable.
            The first level is the lowest level, and the last level is the highest level.

        thresholds : list[float]
            The (upper) thresholds for the levels of the new categorical variable.
            The thresholds must be in ascending order.
            For example, if thresholds = [0, 10, 20],
            and level_names = ["Low", "Medium", "High", "Very High"],
            then the new variable will have the following levels:

            - "Low" for values less than 0,
            - "Medium" for other values less than 10,
            - "High" for other values less than 20,
            - "Very High" for values greater than or equal to 20.

        leq : bool
            Default: False. If True, the thresholds are inclusive.

        Returns
        -------
        Analyzer
            Returns self for method chaining.
        """
        feature_name = rename_var(feature_name)

        if feature_name in self._working_df_train.columns:
            print_wrapped(
                f"Variable {quote_and_color(feature_name, 'purple')} already exists. "
                "Overwriting.",
                type="WARNING",
                level="INFO",
            )

        thresholds_with_infs = [-np.inf] + thresholds + [np.inf]
        if not pd.Index(thresholds_with_infs).is_monotonic_increasing:
            raise ValueError("Thresholds must be monotonic increasing.")

        self._working_df_train[feature_name] = pd.cut(
            self._working_df_train[numeric_var],
            bins=thresholds_with_infs,
            labels=level_names,
            right=leq,
            ordered=False,
        )
        self._working_df_test[feature_name] = pd.cut(
            self._working_df_test[numeric_var],
            bins=thresholds_with_infs,
            labels=level_names,
            right=leq,
            ordered=False,
        )

        # if the level names are all numeric, force new columns to be numeric
        if all(
            [isinstance(name, int) or isinstance(name, float) for name in level_names]
        ):
            self._working_df_train[feature_name] = self._working_df_train[
                feature_name
            ].astype(float)
            self._working_df_test[feature_name] = self._working_df_test[
                feature_name
            ].astype(float)

        if self._verbose:
            print_wrapped(
                f"Engineered categorical variable "
                + quote_and_color(feature_name, "purple")
                + " from numeric variable "
                + quote_and_color(numeric_var, "purple")
                + " with categories "
                + list_to_string(level_names, color="purple")
                + ".",
                type="UPDATE",
            )

        (
            self._categorical_vars,
            self._numeric_vars,
            self._categorical_to_categories,
        ) = self._compute_categorical_numeric_vars(self._working_df_train)

        self._preprocess_step_tracer.add_step(
            "engineer_categorical_feature",
            {
                "feature_name": feature_name,
                "numeric_var": numeric_var,
                "level_names": level_names,
                "thresholds": thresholds,
                "leq": leq,
            },
        )

        return self

    def dropna(
        self,
        include_vars: list[str] | None = None,
        exclude_vars: list[str] | None = None,
    ) -> "DataHandler":
        """Drops rows with missing values in-place on both the working train
        and test DataFrames.

        Parameters
        ----------
        include_vars : list[str]
            Default: None.
            List of columns along which to drop rows with missing values.
            If None, drops rows with missing values in all columns.

        exclude_vars : list[str]
            Default: None.
            List of columns along which to exclude from dropping rows with
            missing values. If None, no variables are excluded.

        Returns
        -------
        DataHandler
            Returns self for method chaining.
        """
        if include_vars is None:
            include_vars = self.vars()
        if exclude_vars is not None:
            include_vars = list(set(include_vars) - set(exclude_vars))

        orig_len_train = self._working_df_train.shape[0]
        orig_len_test = self._working_df_test.shape[0]

        self._working_df_train = self._working_df_train.dropna(subset=include_vars)
        self._working_df_test = self._working_df_test.dropna(subset=include_vars)

        new_len_train = self._working_df_train.shape[0]
        new_len_test = self._working_df_test.shape[0]

        if self._verbose:
            print_wrapped(
                f"Dropped {orig_len_train - new_len_train} rows with missing values "
                + f"from train and {orig_len_test - new_len_test} rows from test.",
                type="UPDATE",
            )

        self._preprocess_step_tracer.add_step(
            "dropna",
            {
                "vars": include_vars,
            },
        )
        return self

    def onehot(
        self,
        include_vars: list[str] | None = None,
        exclude_vars: list[str] | None = None,
        dropfirst: bool = True,
        keep_original: bool = False,
    ) -> "DataHandler":
        """One-hot encodes all categorical variables in-place. Encoder is
        fit on train DataFrame and transforms both train and test DataFrames.

        Parameters
        ----------
        include_vars : list[str]
            Default: None.
            List of categorical variables to one-hot encode. If None, this is set to
            all categorical variables.

        exclude_vars : list[str]
            Default: None.
            List of categorical variables to exclude from one-hot encoding.

        dropfirst : bool
            Default: True.
            If True, the first dummy variable is dropped.

        Returns
        -------
        DataHandler
            Returns self for method chaining.
        """
        if include_vars is None:
            include_vars = self.categorical_vars()
        if exclude_vars is not None:
            include_vars = list(set(include_vars) - set(exclude_vars))

        if len(include_vars) == 0:
            if self._verbose:
                print_wrapped(
                    "No categorical variables found. Skipping one-hot encoding.",
                    type="WARNING",
                )
            return self

        self._working_df_train = self._onehot_helper(
            self._working_df_train,
            vars=include_vars,
            dropfirst=dropfirst,
            fit=True,
            keep_original=keep_original,
        )
        self._working_df_test = self._onehot_helper(
            self._working_df_test,
            vars=include_vars,
            dropfirst=dropfirst,
            fit=False,
            keep_original=keep_original,
        )

        (
            self._categorical_vars,
            self._numeric_vars,
            self._categorical_to_categories,
        ) = self._compute_categorical_numeric_vars(self._working_df_train)

        if self._verbose:
            if dropfirst:
                dropfirst_str = "For each variable, the first category was ignored."
                print_wrapped(
                    f"One-hot encoded {list_to_string(include_vars)}. " + dropfirst_str,
                    type="UPDATE",
                )
            else:
                print_wrapped(
                    f"One-hot encoded {list_to_string(include_vars)}. "
                    + "All categories were encoded",
                    type="UPDATE",
                )

        self._preprocess_step_tracer.add_step(
            "onehot",
            {
                "vars": include_vars,
                "dropfirst": dropfirst,
                "keep_original": keep_original,
            },
        )
        return self

    def drop_highly_missing_vars(
        self,
        include_vars: list[str] | None = None,
        exclude_vars: list[str] | None = None,
        threshold: float = 0.5,
    ) -> "DataHandler":
        """Drops columns with more than a provided percentage of missing values
        (computed on train) in-place for both the working train and test
        DataFrames.

        Parameters
        ----------
        include_vars : list[str] | None
            Default: None. If not None, only drops columns with more than 50% missing
            values in the specified variables. Otherwise, drops columns with more than
            50% missing values in all variables.

        exclude_vars : list[str] | None
            Default: None. If not None, excludes the specified variables from the
            list of variables to drop (which is set to all variables by default).

        threshold : float
            Default: 0.5. Proportion of missing values above which a column is dropped.
            For example, if threshold = 0.2, then columns with more than 20% missing
            values are dropped.

        Returns
        -------
        DataHandler
            Returns self for method chaining.
        """
        if threshold < 0 or threshold > 1:
            raise ValueError("The threshold value must be between 0 and 1.")

        if include_vars is None:
            prev_vars = self._working_df_train.columns.to_list()
        else:
            prev_vars = include_vars

        if exclude_vars is not None:
            prev_vars = sorted(list(set(prev_vars) - set(exclude_vars)))

        missingness = self._working_df_train[prev_vars].isna().mean()
        vars_to_drop = missingness[missingness >= threshold].index.to_list()

        if len(vars_to_drop) == 0:
            print_wrapped(
                f"No variables found with at least {threshold*100}% of values missing.",
                type="WARNING",
            )
            self._preprocess_step_tracer.add_step(
                "drop_highly_missing_vars",
                {
                    "include_vars": include_vars,
                    "exclude_vars": exclude_vars,
                    "threshold": threshold,
                },
            )
            return self

        self._working_df_train = self._working_df_train.drop(vars_to_drop, axis=1)
        self._working_df_test = self._working_df_test.drop(vars_to_drop, axis=1)

        assert (
            self._working_df_test.shape[1] == self._working_df_train.shape[1]
        ), "Train and test DataFrames have different number of columns."

        if self._verbose:
            print_wrapped(
                f"Dropped variables {list_to_string(vars_to_drop)} "
                + f"with at least {threshold*100}% of "
                + "values missing.",
                type="UPDATE",
            )
        (
            self._categorical_vars,
            self._numeric_vars,
            self._categorical_to_categories,
        ) = self._compute_categorical_numeric_vars(self._working_df_train)
        self._preprocess_step_tracer.add_step(
            "drop_highly_missing_vars",
            {
                "include_vars": include_vars,
                "exclude_vars": exclude_vars,
                "threshold": threshold,
            },
        )
        return self

    def impute(
        self,
        include_vars: list[str] | None = None,
        exclude_vars: list[str] | None = None,
        numeric_strategy: Literal["median", "mean", "5nn", "10nn"] = "median",
        categorical_strategy: Literal["most_frequent", "missing"] = "most_frequent",
    ) -> "DataHandler":
        """Imputes missing values in-place. Imputer is fit on train DataFrame
        and transforms both train and test DataFrames.

        Parameters
        ----------
        include_vars : list[str]
            Default: None. List of variables to impute missing values.
            If None, imputes missing values in all columns.

        exclude_vars : list[str]
            Default: None. List of variables to exclude from imputing missing values.
            If None, no variables are excluded.

        numeric_strategy : Literal['median', 'mean', '5nn', '10nn']
            Default: 'median'.
            Strategy for imputing missing values in numeric variables.
            - 'median': impute with median.
            - 'mean': impute with mean.
            - '5nn': impute with 5-nearest neighbors.
            - '10nn': impute with 10-nearest neighbors.

        categorical_strategy : Literal['most_frequent', 'missing']
            Default: 'most_frequent'.
            Strategy for imputing missing values in categorical variables.
            - 'most_frequent': impute with most frequent value.
            - 'missing': impute with a new category 'tm_missing'.

        Returns
        -------
        DataHandler
            Returns self for method chaining.
        """
        numeric_vars = self.numeric_vars()
        categorical_vars = self.categorical_vars()

        if include_vars is not None:
            include_vars_set = set(include_vars)
            numeric_vars = list(include_vars_set & set(numeric_vars))
            categorical_vars = list(include_vars_set & set(categorical_vars))
        if exclude_vars is not None:
            exclude_vars_set = set(exclude_vars)
            numeric_vars = list(set(numeric_vars) - exclude_vars_set)
            categorical_vars = list(set(categorical_vars) - exclude_vars_set)

        numeric_vars = sorted(numeric_vars)
        categorical_vars = sorted(categorical_vars)

        if len(numeric_vars) == 0 and len(categorical_vars) == 0:
            raise ValueError("No variables were provided.")

        # keep only the vars with missing values
        df = self.df_all()
        missing_numeric_vars = [var for var in numeric_vars if df[var].isna().any()]
        missing_categorical_vars = [
            var for var in categorical_vars if df[var].isna().any()
        ]

        if len(missing_numeric_vars) != len(numeric_vars):
            vars_not_missing = list(set(numeric_vars) - set(missing_numeric_vars))
            print_wrapped(
                f"Numeric variables {list_to_string(vars_not_missing)} "
                + "have no missing values. "
                + "Imputer will consider all specified variables regardless.",
                type="NOTE",
            )

        if len(missing_categorical_vars) != len(categorical_vars):
            vars_not_missing = list(
                set(categorical_vars) - set(missing_categorical_vars)
            )
            print_wrapped(
                f"Categorical variables {list_to_string(vars_not_missing)} "
                + "have no missing values. "
                + "Imputer will consider all specified variables regardless.",
                type="NOTE",
            )

        # impute numeric variables
        if len(numeric_vars) > 0:
            if numeric_strategy == "5nn":
                imputer = KNNImputer(n_neighbors=5, keep_empty_features=True)
            elif numeric_strategy == "10nn":
                imputer = KNNImputer(n_neighbors=10, keep_empty_features=True)
            elif numeric_strategy in ["median", "mean"]:
                imputer = SimpleImputer(
                    strategy=numeric_strategy, keep_empty_features=True
                )
            else:
                raise ValueError("Invalid numeric imputation strategy.")
            self._working_df_train[numeric_vars] = imputer.fit_transform(
                self._working_df_train[numeric_vars]
            )
            self._working_df_test[numeric_vars] = imputer.transform(
                self._working_df_test[numeric_vars]
            )

        # impute categorical variables
        if len(categorical_vars) > 0:
            if categorical_strategy == "missing":
                imputer = SimpleImputer(
                    strategy="constant",
                    fill_value="tm_missing",
                    keep_empty_features=True,
                )
            elif categorical_strategy == "most_frequent":
                imputer = SimpleImputer(
                    strategy="most_frequent", keep_empty_features=True
                )
            else:
                raise ValueError("Invalid categorical imputation strategy.")
            self._working_df_train[categorical_vars] = imputer.fit_transform(
                self._working_df_train[categorical_vars]
            )
            self._working_df_test[categorical_vars] = imputer.transform(
                self._working_df_test[categorical_vars]
            )

        if self._verbose:
            message = "Imputed missing values with "

            if len(numeric_vars) > 0:
                message += (
                    f"strategy {quote_and_color(numeric_strategy, 'yellow')} "
                    + f"for numeric variables {list_to_string(numeric_vars)}"
                )
                if len(categorical_vars) > 0:
                    message += " and "

            if len(categorical_vars) > 0:
                message += (
                    f"strategy {quote_and_color(categorical_strategy, 'yellow')} "
                    + f"for categorical variables {list_to_string(categorical_vars)}"
                )

            message += "."

            print_wrapped(
                message,
                type="UPDATE",
            )

        self._preprocess_step_tracer.add_step(
            "impute",
            {
                "vars": numeric_vars + categorical_vars,
                "numeric_strategy": numeric_strategy,
                "categorical_strategy": categorical_strategy,
            },
        )

        return self

    def scale(
        self,
        include_vars: list[str] | None = None,
        exclude_vars: list[str] | None = None,
        strategy: Literal[
            "standardize",
            "minmax",
            "log",
            "log1p",
            "robust_standardize",
            "normal_quantile",
            "uniform_quantile",
        ] = "standardize",
    ) -> "DataHandler":
        """Scales variable values.

        Parameters
        ----------
        include_vars : list[str]
            Default: None. List of variables to scale.
            If None, scales values in all columns.

        exclude_vars : list[str]
            Default: None. List of variables to exclude from scaling.
            If None, no variables are excluded.

        strategy : str
            Name of the scaling strategy.

        Returns
        -------
        DataHandler
            Returns self for method chaining.
        """
        if include_vars is None:
            include_vars = self.numeric_vars()
        if exclude_vars is not None:
            include_vars = list(set(include_vars) - set(exclude_vars))

        for var in include_vars:
            if var not in self._numeric_vars:
                print_wrapped(
                    f"Variable {var} is not numeric. Skipping.", type="WARNING"
                )
                continue

            train_data = self._working_df_train[var].to_numpy()
            if strategy == "standardize":
                scaler = StandardizeSingleVar(var, train_data)
            elif strategy == "minmax":
                scaler = MinMaxSingleVar(var, train_data)
            elif strategy == "log":
                scaler = LogTransformSingleVar(var, train_data)
            elif strategy == "log1p":
                scaler = Log1PTransformSingleVar(var, train_data)
            elif strategy == "robust_standardize":
                scaler = RobustStandardizeSingleVar(var, train_data)
            elif strategy == "normal_quantile":
                scaler = NormalQuantileTransformSingleVar(var, train_data)
            elif strategy == "uniform_quantile":
                scaler = UniformQuantileTransformSingleVar(var, train_data)
            else:
                raise ValueError("Invalid scaling strategy.")

            self._working_df_train[var] = scaler.transform(
                self._working_df_train[var].to_numpy()
            )
            self._working_df_test[var] = scaler.transform(
                self._working_df_test[var].to_numpy()
            )
            if var not in self._numeric_var_to_scalers:
                self._numeric_var_to_scalers[var] = [scaler]
            else:
                self._numeric_var_to_scalers[var].append(scaler)

        if self._verbose:
            print_wrapped(
                f"Scaled variables {list_to_string(include_vars)} "
                + f'using strategy {quote_and_color(strategy, "yellow")}.',
                type="UPDATE",
            )

        self._preprocess_step_tracer.add_step(
            "scale", {"vars": include_vars, "strategy": strategy}
        )
        return self

    def add_scaler(self, scaler: BaseSingleVarScaler, var: str) -> "DataHandler":
        """Adds a scaler for the given variable. Does not scale the variable.

        Parameters
        ----------
        scaler : BaseSingleVarScaler
            Scaler object.

        var : str
            Name of the variable.

        Returns
        -------
        DataHandler
            Returns self.
        """
        if var not in self._numeric_vars:
            raise ValueError(
                f"Variable {var} is not found in the set of numeric variables."
            )
        self._numeric_var_to_scalers[var].append(scaler)
        self._preprocess_step_tracer.add_step(
            "add_scaler", {"scaler": scaler, "var": var}
        )
        return self

    def select_vars(self, vars: list[str]) -> "DataHandler":
        """Selects subset of (column) variables in-place on the working
        train and test DataFrames.

        Parameters
        ----------
        vars : list[str]
            List of variable names.

        Returns
        -------
        DataHandler
            Returns self.
        """
        self._working_df_test = self._working_df_test[vars]
        self._working_df_train = self._working_df_train[vars]
        (
            self._categorical_vars,
            self._numeric_vars,
            self._categorical_to_categories,
        ) = self._compute_categorical_numeric_vars(self._working_df_train)
        if self._verbose:
            shapes_dict = self._shapes_str_formatted()
            print_wrapped(
                f"Selected columns {list_to_string(vars)}. "
                + "Shapes of train, test DataFrames: "
                + f'{shapes_dict["train"]}, {shapes_dict["test"]}.',
                type="UPDATE",
            )

        self._preprocess_step_tracer.add_step("select_vars", {"vars": vars})

        return self

    def drop_vars(self, vars: list[str]) -> "DataHandler":
        """Drops subset of variables (columns) in-place on the working
        train and test DataFrames.

        Parameters
        ----------
        vars : list[str]

        Returns
        -------
        DataHandler
            Returns self for method chaining.
        """
        self._working_df_test = self._working_df_test.drop(vars, axis="columns")
        self._working_df_train = self._working_df_train.drop(vars, axis="columns")
        (
            self._categorical_vars,
            self._numeric_vars,
            self._categorical_to_categories,
        ) = self._compute_categorical_numeric_vars(self._working_df_train)
        if self._verbose:
            shapes_dict = self._shapes_str_formatted()
            print_wrapped(
                f"Dropped columns {list_to_string(vars)}. "
                + "Shapes of train, test DataFrames: "
                + f'{shapes_dict["train"]}, {shapes_dict["test"]}.',
                type="UPDATE",
            )

        self._preprocess_step_tracer.add_step("drop_vars", {"vars": vars})

        return self

    def force_numeric(self, vars: list[str]) -> "DataHandler":
        """Forces variables to numeric (floats).

        Parameters
        ----------
        vars : list[str]
            Name of variables to force to numeric.

        Returns
        -------
        DataHandler
            Returns self for method chaining.
        """
        successfully_forced = []

        for var in vars:
            if var not in self._working_df_train.columns:
                raise ValueError(f"Invalid variable name: {var}.")
            try:
                self._working_df_train[var] = self._working_df_train[var].apply(
                    lambda x: float(x) if pd.notna(x) else np.nan
                )
                self._working_df_test[var] = self._working_df_test[var].apply(
                    lambda x: float(x) if pd.notna(x) else np.nan
                )
            except Exception:
                if self._verbose:
                    print_wrapped(
                        "Unable to force variable "
                        + f"{quote_and_color(var)} to numeric.",
                        type="WARNING",
                    )

            successfully_forced.append(var)

        if self._verbose:
            print_wrapped(
                f"Forced variables {list_to_string(successfully_forced)} "
                + "to numeric.",
                type="UPDATE",
            )

        self._preprocess_step_tracer.add_step("force_numeric", {"vars": vars})

        return self

    def force_binary(
        self,
        vars: list[str],
        pos_labels: list[str] | None = None,
        ignore_multiclass: bool = False,
        rename: bool = False,
    ) -> "DataHandler":
        """Forces variables to be binary (0 and 1 valued numeric variables).
        Does nothing if the data contains more than two classes unless
        ignore_multiclass is True and pos_label is specified,
        in which case all classes except pos_label are labeled with zero.

        Parameters
        ----------
        vars : list[str]
            Name of variables to force to binary.

        pos_labels : list[str]
            Default: None. The positive labels.
            If None, the first class for each var is the positive label.

        ignore_multiclass : bool
            Default: False. If True, all classes except pos_label are labeled with
            zero. Otherwise raises ValueError.

        rename : bool
            Default: False. If True, the variables are renamed to
            {var}::{pos_label}.

        Returns
        -------
        DataHandler
            Returns self for method chaining.
        """
        if pos_labels is None and ignore_multiclass:
            raise ValueError(
                "pos_labels must be specified if ignore_multiclass is True."
            )

        if not isinstance(vars, list):
            raise ValueError("vars must be lists.")
        if pos_labels is not None and not isinstance(pos_labels, list):
            raise ValueError("pos_labels must be lists.")

        vars_to_renamed = {}
        for i, var in enumerate(vars):
            if var not in self._working_df_train.columns:
                raise ValueError(f"Invalid variable name: {var}.")

            if pos_labels is None:
                unique_vals = self._working_df_train[var].unique()
                if len(unique_vals) > 2:
                    if self._verbose:
                        print_wrapped(
                            "More than two classes present for "
                            + f"{var}. Skipping {var}.",
                            type="WARNING",
                        )
                    continue
                pos_label = unique_vals[0]
                self._working_df_train[var] = self._working_df_train[var].apply(
                    lambda x: 1 if x == pos_label else 0
                )
                self._working_df_test[var] = self._working_df_test[var].apply(
                    lambda x: 1 if x == pos_label else 0
                )
            else:
                unique_vals = self._working_df_train[var].unique()
                if len(unique_vals) > 2:
                    if not ignore_multiclass:
                        if self._verbose:
                            print_wrapped(
                                "More than two classes present for "
                                + f"{var}. Skipping {var}.",
                                type="WARNING",
                            )
                        continue

                pos_label = pos_labels[i]
                self._working_df_train[var] = self._working_df_train[var].apply(
                    lambda x: 1 if x == pos_label else 0
                )
                self._working_df_test[var] = self._working_df_test[var].apply(
                    lambda x: 1 if x == pos_label else 0
                )

            vars_to_renamed[var] = f"{var}::{pos_label}"

        if rename:
            self._working_df_train = self._working_df_train.rename(
                columns=vars_to_renamed
            )
            self._working_df_test = self._working_df_test.rename(
                columns=vars_to_renamed
            )

        if self._verbose:
            if len(vars_to_renamed) == 0:
                print_wrapped("No variables were forced to binary.", type="WARNING")
            else:
                old_vars_txt = color_text(
                    list_to_string(vars_to_renamed.keys()), "purple"
                )
                new_vars_txt = color_text(
                    list_to_string(vars_to_renamed.values()), "purple"
                )
                if rename:
                    print_wrapped(
                        f"Forced variables {old_vars_txt} to binary. "
                        + f"Variables renamed to {new_vars_txt}.",
                        type="UPDATE",
                    )
                else:
                    print_wrapped(
                        f"Forced variables {old_vars_txt} to binary.",
                        type="UPDATE",
                    )

        (
            self._categorical_vars,
            self._numeric_vars,
            self._categorical_to_categories,
        ) = self._compute_categorical_numeric_vars(self._working_df_train)
        self._preprocess_step_tracer.add_step(
            "force_binary",
            {
                "vars": vars,
                "pos_labels": pos_labels,
                "ignore_multiclass": ignore_multiclass,
                "rename": rename,
            },
        )
        return self

    def force_categorical(self, vars: list[str]) -> "DataHandler":
        """Forces variables to become categorical. That is, converts the
        variables to string dtype.

        Parameters
        ----------
        vars : list[str]

        Returns
        -------
        DataHandler
            Returns self for method chaining.
        """
        if not isinstance(vars, list):
            vars = [vars]

        for var in vars:
            self._working_df_train[var] = self._working_df_train[var].apply(
                lambda x: str(x) if pd.notna(x) else np.nan
            )
            self._working_df_test[var] = self._working_df_test[var].apply(
                lambda x: str(x) if pd.notna(x) else np.nan
            )

        if self._verbose:
            print_wrapped(
                f"Forced variables {list_to_string(vars)} to categorical.",
                type="UPDATE",
            )

        (
            self._categorical_vars,
            self._numeric_vars,
            self._categorical_to_categories,
        ) = self._compute_categorical_numeric_vars(self._working_df_train)

        self._preprocess_step_tracer.add_step("force_categorical", {"vars": vars})

        return self

    # --------------------------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------------------------
    def _verify_input_dfs(self, df1: pd.DataFrame, df2: pd.DataFrame):
        """Ensures that the original train and test DataFrames have the
        same variables.

        Parameters
        ----------
        df1 : pd.DataFrame
            The train DataFrame.

        df2 : pd.DataFrame
            The test DataFrame.

        Raises
        ------
        ValueError
        """
        l1 = df1.columns.to_list()
        l2 = df2.columns.to_list()
        vars_not_in_both = list(set(l1) ^ set(l2))
        if len(vars_not_in_both) > 0:
            raise ValueError(
                f"Variables {list_to_string(vars_not_in_both)} "
                + "are not in both train and test DataFrames."
            )
        datetime_columns = df1.select_dtypes(
            include=["datetime", np.datetime64]
        ).columns.tolist()
        if len(datetime_columns) > 0:
            raise ValueError(
                f"Variables {list_to_string(datetime_columns)} "
                "are of type datetime. TableMage cannot handle datetime values."
            )

    def _compute_categorical_numeric_vars(
        self, df: pd.DataFrame
    ) -> tuple[list[str], list[str], dict]:
        """Returns the categorical and numeric column values.
        Also returns the categorical variables mapped to their categories.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        list[str]
            List of categorical variables.

        list[str]
            List of numeric variables.

        dict
            Dictionary mapping categorical variables to their categories.
        """
        categorical_vars = df.select_dtypes(
            include=["category", "object"]
        ).columns.to_list()

        numeric_vars = df.select_dtypes(include=["number"]).columns.to_list()

        # force categorical vars to be strings
        for var in categorical_vars:
            df[var] = df[var].apply(lambda x: str(x) if pd.notna(x) else np.nan)

        all_vars = df.columns.to_list()

        # identify vars not in both but in all_vars
        vars_not_in_both = list(set(all_vars) - set(categorical_vars + numeric_vars))

        if len(vars_not_in_both) > 0:
            var_to_type = {var: df[var].dtype for var in vars_not_in_both}
            raise ValueError(
                f"Variables {list_to_string(vars_not_in_both)} "
                + f"are of incompatible types: {str(var_to_type)}"
            )

        categorical_mapped = self._compute_categories(df, categorical_vars)
        return sorted(categorical_vars), sorted(numeric_vars), categorical_mapped

    def _force_train_test_var_agreement(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Modifies df_test to have the same columns as df_train. This helps
        mitigate any problems that may arise from one-hot-encoding the test set.

        Parameters
        ----------
        df_train : pd.DataFrame

        df_test : pd.DataFrane

        Returns
        -------
        pd.DataFrame
            Modified train DataFrame.

        pd.DataFrame
            Modified test DataFrame.
        """
        missing_test_columns = list(set(df_train.columns) - set(df_test.columns))
        extra_test_columns = list(set(df_test.columns) - set(df_train.columns))
        if len(extra_test_columns) > 0:
            if self._verbose:
                print_wrapped(
                    f"Columns {list_to_string(extra_test_columns)} not "
                    + "in train have been dropped from test.",
                    type="WARNING",
                )
            df_test = df_test.drop(columns=extra_test_columns, axis=1)
        if len(missing_test_columns) > 0:
            if self._verbose:
                print_wrapped(
                    f"Columns {list_to_string(missing_test_columns)} not "
                    + "in test have been added to test with 0-valued entries.",
                    type="WARNING",
                )
            for col in missing_test_columns:
                df_test[col] = 0
        assert len(df_test.columns) == len(df_train.columns)

        # ensure that the train and test dfs have the same order
        # (for nicer exploration)
        df_test = df_test[df_train.columns]
        for a, b in zip(df_test.columns, df_train.columns):
            assert a == b

        return df_train, df_test

    def _rename_varnames(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame, silent: bool = False
    ):
        """Renames variables to remove 'problematic' characters. We allow
        underscores, colons, semicolons, and parentheses in variable names,
        in addition to alphanumeric characters. We replace other characters with
        underscores.

        We remove the following characters from variable names:
            '.', '[', ']', '{', '}', '\\', '|', '&',
            '%', '$', '#', '\\n', '\\t'

        Parameters
        ----------
        df_train : pd.DataFrame
            The train DataFrame.

        df_test : pd.DataFrame
            The test DataFrame.

        Returns
        -------
        pd.DataFrame
            The train DataFrame with renamed variable names.

        pd.DataFrame
            The test DataFrame with renamed variable names.
        """
        curr_vars = df_train.columns.to_list()
        curr_to_new = rename_vars(curr_vars)
        new_columns = [curr_to_new[var] for var in curr_vars]
        orig_renamed_vars = [var for var in curr_vars if var != curr_to_new[var]]
        renamed_vars = [curr_to_new[var] for var in orig_renamed_vars]
        df_train.columns = new_columns
        df_test.columns = new_columns
        if self._verbose and not silent:
            if len(renamed_vars) > 0:
                print_wrapped(
                    f"Renamed variables {list_to_string(orig_renamed_vars)} "
                    + f"to {list_to_string(renamed_vars)}.",
                    type="UPDATE",
                )
        return df_train, df_test

    def _shapes_str_formatted(self) -> dict:
        """Returns a dictionary containing shape information for the
        working DataFrames.

        Returns
        -------
        dict
            {
                "train": shape of working_df_train,
                "test": shape of working_df_test
            }
        """
        return {
            "train": color_text(str(self._working_df_train.shape), color="yellow"),
            "test": color_text(str(self._working_df_test.shape), color="yellow"),
        }

    def _compute_categories(
        self, df: pd.DataFrame, categorical_vars: list[str]
    ) -> dict:
        """Returns a dictionary containing the categorical variables
        each mapped to a list of all categories in the variable.

        Parameters
        ----------
        df : pd.DataFrame

        categorical_vars : list[str]

        Returns
        -------
        dict
        """
        categories_dict = {}
        for var in categorical_vars:
            categories_dict[var] = df[var].unique().tolist()
        return categories_dict

    def _onehot_helper(
        self,
        df: pd.DataFrame,
        vars: list[str] | None = None,
        dropfirst: bool = True,
        fit: bool = True,
        keep_original: bool = False,
    ) -> pd.DataFrame:
        """One-hot encodes all categorical variables with more than
        two categories.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to one-hot encode.

        vars : list[str]
            Default: None. If not None, only one-hot encodes the specified variables.

        dropfirst : bool
            Default: True. If True, the first dummy variable is dropped.

        fit : bool
            Default: True.
            If True, fits the encoder on the training data. Otherwise,
            only transforms the test data.

        keep_original : bool
            Default: False.
            If True, keeps the original categorical variable in the DataFrame.

        Returns
        -------
        pd.DataFrame
            The DataFrame with one-hot encoded variables.
        """
        if vars is None:
            categorical_vars, _, _ = self._compute_categorical_numeric_vars(df)
        else:
            for var in vars:
                if var not in df.columns:
                    raise ValueError(f"Invalid variable name: {var}")
            categorical_vars = vars

        if categorical_vars:
            if dropfirst:
                drop = "first"
            else:
                drop = "if_binary"

            if fit:
                self._onehot_encoder = CustomOneHotEncoder(
                    drop=drop, sparse_output=False, handle_unknown="ignore"
                )
                encoded = self._onehot_encoder.fit_transform(df[categorical_vars])
                feature_names = self._onehot_encoder.get_feature_names_out(
                    categorical_vars
                )
                df_encoded = pd.DataFrame(
                    encoded, columns=feature_names, index=df.index
                )

            else:
                encoded = ignore_warnings(self._onehot_encoder.transform)(
                    df[categorical_vars]
                )
                feature_names = self._onehot_encoder.get_feature_names_out(
                    categorical_vars
                )
                df_encoded = pd.DataFrame(
                    encoded, columns=feature_names, index=df.index
                )

            # for all columns in df_encoded, rename
            curr_vars = df_encoded.columns.to_list()
            curr_to_new = rename_vars(curr_vars)
            new_columns = [curr_to_new[var] for var in curr_vars]
            df_encoded.columns = new_columns

            if keep_original:
                return pd.concat([df_encoded, df], axis=1)
            else:
                return pd.concat(
                    [df_encoded, df.drop(columns=categorical_vars)], axis=1
                )
        else:
            return df

    def __len__(self) -> int:
        """Returns the number of examples in working_df_train."""
        return len(self._working_df_train)

    def __str__(self) -> str:
        """Returns a string representation of the DataHandler object."""
        working_df_test = self._working_df_test
        working_df_train = self._working_df_train

        max_width = print_options._max_line_width

        textlen_shapes = (
            len(str(working_df_train.shape) + str(working_df_test.shape)) + 25
        )
        shapes_message_buffer_left = (max_width - textlen_shapes) // 2
        shapes_message_buffer_right = int(np.ceil((max_width - textlen_shapes) / 2))

        shapes_message = (
            color_text(bold_text("Train shape: "), "none")
            + color_text(str(working_df_train.shape), "yellow")
            + " " * shapes_message_buffer_left
            + color_text(bold_text("Test shape: "), "none")
            + color_text(str(working_df_test.shape), "yellow")
            + " " * shapes_message_buffer_right
        )

        title_message = color_text(bold_text(self._name), "none")
        title_message = fill_ignore_format(title_message, width=max_width)

        categorical_message = (
            color_text(bold_text("Categorical variables:"), "none") + "\n"
        )

        categorical_vars = self.categorical_vars()
        categorical_var_message = ""
        if len(categorical_vars) == 0:
            categorical_var_message += color_text("None", "yellow")
        else:
            categorical_var_message += list_to_string(categorical_vars)
        categorical_var_message = fill_ignore_format(
            categorical_var_message,
            width=max_width,
            initial_indent=2,
            subsequent_indent=2,
        )

        numeric_message = color_text(bold_text("Numeric variables:"), "none") + "\n"

        numeric_vars = self.numeric_vars()
        numeric_var_message = ""
        if len(numeric_vars) == 0:
            numeric_var_message += color_text("None", "yellow")
        else:
            numeric_var_message += list_to_string(numeric_vars)
        numeric_var_message = fill_ignore_format(
            numeric_var_message,
            width=max_width,
            initial_indent=2,
            subsequent_indent=2,
        )

        bottom_divider = "\n" + color_text("=" * max_width, "none")
        divider = "\n" + color_text("-" * max_width, "none") + "\n"
        divider_invisible = "\n" + " " * max_width + "\n"
        top_divider = color_text("=" * max_width, "none") + "\n"

        final_message = (
            top_divider
            + title_message
            + divider
            + shapes_message
            + divider
            + categorical_message
            + categorical_var_message
            + divider_invisible
            + numeric_message
            + numeric_var_message
            + bottom_divider
        )

        return final_message

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def copy(self) -> "DataHandler":
        """Returns a copy of the DataHandler object."""
        return deepcopy(self)
