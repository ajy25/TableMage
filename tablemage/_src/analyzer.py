import pandas as pd
from typing import Literal
from sklearn.model_selection import train_test_split
from .ml.predict import (
    BaseR,
    MLRegressionReport,
    BaseC,
    MLClassificationReport,
)
from .ml.cluster import BaseClust, ClusterReport
from .feature_selection import BaseFSR, BaseFSC, VotingSelectionReport
from .linear import (
    OLSLinearModel,
    OLSReport,
    LogitLinearModel,
    LogitReport,
    MNLogitLinearModel,
    MNLogitReport,
    parse_and_transform_rlike,
)
from .exploratory import (
    EDAReport,
)
from .causal import CausalModel
from .display.print_utils import print_wrapped, quote_and_color
from .data.datahandler import DataHandler
from .utils import ensure_arg_list_uniqueness


class Analyzer:
    """Analyzer is a class designed for conducting exploratory data analysis (EDA),
    regression analysis, and machine learning modeling on wide format tabular data.

    An Analyzer object can be initialized from a single DataFrame which is then
    split into train and test DataFrames, or, alternatively, from pre-split
    train and test DataFrames. The object can then be used to conduct
    a variety of analyses,
    including exploratory data analysis (the eda() method),
    regression analysis (ols() and logit() methods),
    and machine learning modeling (classify() and regress() methods).

    The Analyzer object also handles data preprocessing tasks, such as scaling,
    imputing missing values, dropping rows with missing values, one-hot encoding,
    and selecting variables. These methods can be chained together for easy data
    transformation. The Analyzer object remembers how the data was transformed,
    enabling proper fitting and transforming of cross validation splits of the
    train dataset.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        df_test: pd.DataFrame | None = None,
        test_size: float = 0.0,
        split_seed: int = 42,
        id_column: str | None = None,
        verbose: bool = True,
        name: str = "Unnamed Dataset",
    ):
        """Initializes a Analyzer object.

        Parameters
        ----------
        df : pd.DataFrame | None
            The DataFrame to be analyzed. Must be in wide format, i.e. with shape
            (n_units, n_vars). If df_test is provided, then the df is treated as the
            train DataFrame. Otherwise, the df is split into train and test DataFrames
            according to the test_size parameter.

        df_test : pd.DataFrame | None
            Default: None.
            If not None, then treats df as the train DataFrame.

        test_size : float
            Default: 0. Proportion of the DataFrame to withhold for
            testing. If test_size = 0, then the train DataFrame and the
            test DataFrame will both be the same as the input df.
            If df_test is provided, then test_size is ignored.

        id_column : str | None
            Default: None. The name of the column containing unique identifiers.
            If not None, then the column will be set as the index of the DataFrame.
            If None, then the input index will be used as the index of the DataFrame.

        split_seed : int
            Default: 42.
            Used only for the train test split.
            If df_test is provided, then split_seed is ignored.

        verbose : bool
            Default: False. If True, prints helpful update messages for certain
            Analyzer function calls.

        name : str
            Default: 'Unnamed Dataset'. Name of the dataset the Analyzer is
            initialized for.
        """

        self._verbose = verbose

        # force column names to str
        df.columns = df.columns.astype(str)

        if df_test is not None:
            df_test.columns = df_test.columns.astype(str)

            # ensure column names are sorted
            df = df.reindex(sorted(df.columns), axis=1)
            df_test = df_test.reindex(sorted(df_test.columns), axis=1)

            if id_column is not None:
                if id_column not in df.columns:
                    raise ValueError(f"ID column {id_column} not found in train data.")
                if id_column not in df_test.columns:
                    raise ValueError(f"ID column {id_column} not found in test data.")
                df = df.set_index(id_column)
                df_test = df_test.set_index(id_column)

            self._datahandler = DataHandler(
                df_train=df, df_test=df_test, verbose=self._verbose, name=name
            )

        else:
            if test_size > 0:
                temp_train, temp_test = train_test_split(
                    df, test_size=test_size, shuffle=True, random_state=split_seed
                )
                temp_train_df = pd.DataFrame(temp_train, columns=df.columns)
                temp_test_df = pd.DataFrame(temp_test, columns=df.columns)
            else:
                if self._verbose:
                    print_wrapped(
                        "No test dataset provided. The test dataset "
                        + "will be treated as a train dataset copy.",
                        type="NOTE",
                    )
                temp_train_df = df
                temp_test_df = df
            # ensure column names are sorted
            temp_train_df = temp_train_df.reindex(sorted(temp_train_df.columns), axis=1)
            temp_test_df = temp_test_df.reindex(sorted(temp_test_df.columns), axis=1)

            if id_column is not None:
                if id_column not in temp_train_df.columns:
                    raise ValueError(f"ID column {id_column} not found in train data.")
                if id_column not in temp_test_df.columns:
                    raise ValueError(f"ID column {id_column} not found in test data.")
                temp_train_df = temp_train_df.set_index(id_column)
                temp_test_df = temp_test_df.set_index(id_column)

            self._datahandler = DataHandler(
                df_train=temp_train_df,
                df_test=temp_test_df,
                verbose=self._verbose,
                name=name,
            )

        self._name = name

        if len(self._datahandler.df_train()) == 0:
            raise ValueError("Train DataFrame is empty.")
        if len(self._datahandler.df_test()) == 0:
            raise ValueError("Test DataFrame is empty.")

        if self._verbose:
            print_wrapped(
                "Analyzer initialized for dataset "
                f"{quote_and_color(self._name, 'yellow')}.",
                type="UPDATE",
            )

    # --------------------------------------------------------------------------
    # EDA + FEATURE SELECTION + CAUSAL EFFECT ESTIMATION + REGRESSION ANALYSIS
    # --------------------------------------------------------------------------
    def eda(self, dataset: Literal["train", "test", "all"] = "all") -> EDAReport:
        """Constructs an EDAReport object for the working train
        DataFrame, the working test DataFrame, or both DataFrames combined.

        Parameters
        ----------
        dataset : Literal['train', 'test', 'all']
            The dataset to be analyzed. By default, analyzes all data.

        Returns
        -------
        EDAReport
            The EDAReport object contains a variety of exploratory data
            analysis methods, including summary statistics for numeric and
            categorical variables, t-tests, and data visualizations.
        """
        if dataset == "train":
            return EDAReport(self._datahandler.df_train())
        elif dataset == "test":
            return EDAReport(self._datahandler.df_test())
        elif dataset == "all":
            return EDAReport(self._datahandler.df_all())
        else:
            raise ValueError(f"Invalid input: dataset = {dataset}.")

    @ensure_arg_list_uniqueness()
    def causal(
        self,
        treatment: str,
        outcome: str,
        confounders: list[str],
        dataset: Literal["train", "test", "all"] = "all",
    ) -> CausalModel:
        """Returns a CausalModel object for estimating causal effects.

        Parameters
        ----------
        treatment : str
            The treatment variable.

        outcome : str
            The outcome variable.

        confounders : list[str]
            The confounding variables.

        dataset : Literal['train', 'test', 'all']
            The dataset to be analyzed. By default, analyzes all data.

        Returns
        -------
        CausalModel
            The CausalModel object contains methods for estimating causal effects.
        """
        return CausalModel(
            datahandler=self._datahandler,
            treatment=treatment,
            outcome=outcome,
            confounders=confounders,
            dataset=dataset,
        )

    @ensure_arg_list_uniqueness()
    def select_features(
        self,
        target: str,
        predictors: list[str] | None = None,
        feature_selectors: list[BaseFSR] | list[BaseFSC] | None = None,
        max_n_features: int | None = None,
    ) -> VotingSelectionReport:
        """Selects the most important features using a variety of feature selection
        methods. The feature selection methods can be used to select the most
        important predictors for regression or classification.

        Parameters
        ----------
        target : str
            The target variable.

        predictors : list[str] | None
            Default: None. The predictors to select from.
            If None, uses all variables except the target as predictors.

        feature_selectors : list[BaseFSR] | list[BaseFSC] | None
            Default: None. The feature selection methods to use.
            If None, uses all feature selection methods.

        max_n_features : int | None
            Default: None. Maximum number of features to select.
            If None, then all features with at least 50% support are selected.

        Returns
        -------
        VotingSelectionReport
            Report object containing the results of the feature selection methods.
        """
        if target in self._datahandler.categorical_vars():
            for fs in feature_selectors:
                if not isinstance(fs, BaseFSC):
                    raise ValueError(
                        "Feature selection methods for classification "
                        + "should be instances of BaseFSC."
                    )

        if predictors is None:
            predictors = self._datahandler.vars()
            if target in predictors:
                predictors.remove(target)

        for predictor in predictors:
            if predictor not in self._datahandler.vars():
                raise ValueError(f"Predictor {predictor} not found in data.")

        return VotingSelectionReport(
            selectors=feature_selectors,
            dataemitter=self._datahandler.train_test_emitter(
                y_var=target,
                X_vars=predictors,
            ),
            max_n_features=max_n_features,
        )

    @ensure_arg_list_uniqueness()
    def ols(
        self,
        target: str | None = None,
        predictors: list[str] | None = None,
        formula: str | None = None,
        alpha: float = 0.0,
        l1_weight: float = 0.0,
    ) -> OLSReport:
        """Performs OLS regression.
        If formula is provided, performs regression with OLS via formula.
        Units with missing data will be dropped.

        Parameters
        ----------
        target : str | None
            Default: None. The variable to be predicted.

        predictors : list[str]
            Default: None.
            If None, all variables except target will be used as predictors.

        formula : str | None
            Default: None. If not None, uses formula to specify the regression
            (overrides target and predictors).

        alpha : float
            Default: 0. Regularization strength. Must be a positive float.

        l1_weight : float
            Default: 0. The weight of the L1 penalty. Must be a float between 0 and 1.

        Returns
        -------
        OLSReport
            The OLSReport object contains a variety of OLS regression methods,
            including summary statistics, model coefficients, and data visualizations.
        """
        if formula is None and target is None:
            raise ValueError("target must be specified if formula is None.")

        elif formula is None:
            if target not in self._datahandler.numeric_vars():
                raise ValueError(
                    f"Target variable {quote_and_color(target, 'yellow')} "
                    + "is not numeric."
                )
            if predictors is None:
                predictors = self._datahandler.vars()
                if target in predictors:
                    if self._verbose:
                        print_wrapped(
                            f"Removing target variable {quote_and_color(target, 'yellow')} "
                            + "from predictors.",
                            type="WARNING",
                        )
                    predictors.remove(target)
            return OLSReport(
                OLSLinearModel(alpha=alpha, l1_weight=l1_weight),
                self._datahandler,
                target,
                predictors,
            )

        else:
            try:
                y_series_train, y_scaler, X_df_train = parse_and_transform_rlike(
                    formula, self._datahandler.df_train()
                )
                y_series_test, _, X_df_test = parse_and_transform_rlike(
                    formula, self._datahandler.df_test()
                )
            except Exception as e:
                raise ValueError(f"Invalid formula: {formula}. Error: {e}.")

            # ensure missing values are dropped
            y_X_df_combined_train = pd.DataFrame(y_series_train).join(X_df_train)
            y_X_df_combined_test = pd.DataFrame(y_series_test).join(X_df_test)
            y_X_df_combined_train = y_X_df_combined_train.dropna()
            y_X_df_combined_test = y_X_df_combined_test.dropna()
            (
                y_X_df_combined_train,
                y_X_df_combined_test,
            ) = self._datahandler._force_train_test_var_agreement(
                y_X_df_combined_train, y_X_df_combined_test
            )

            predictors = y_X_df_combined_train.columns.to_list()
            target = y_series_train.name
            predictors.remove(target)

            if target not in self._datahandler.numeric_vars():
                raise ValueError(
                    f"Target variable {quote_and_color(target, 'yellow')} "
                    + "is not numeric."
                )

            datahandler = DataHandler(
                y_X_df_combined_train, y_X_df_combined_test, verbose=False
            )

            if y_scaler is not None:
                datahandler.add_scaler(y_scaler, target)
            elif self._datahandler.scaler(target) is not None:
                datahandler.add_scaler(self._datahandler.scaler(target), target)

            return OLSReport(
                OLSLinearModel(alpha=alpha, l1_weight=l1_weight),
                datahandler,
                target,
                predictors,
            )

    @ensure_arg_list_uniqueness()
    def logit(
        self,
        target: str | None = None,
        predictors: list[str] | None = None,
        formula: str | None = None,
        alpha: float = 0.0,
        l1_weight: float = 0.0,
        threshold_strategy: Literal["f1", "roc"] | None = None,
    ) -> LogitReport | MNLogitReport:
        """Performs logistic regression.
        if formula is provided, performs logistic regression via formula.
        Units with missing data will be dropped.

        Parameters
        ----------
        target : str | None
            Default: None. The variable to be predicted.

        predictors : list[str] | None
            Default: None.
            If None, all variables except target will be used as predictors.

        formula : str | None
            Default: None. If not None, uses formula to specify the regression
            (overrides target and predictors).

        alpha : float
            Default: 0. Regularization strength. Must be a positive float.

        l1_weight : float
            Default: 0. The weight of the L1 penalty. Must be a float between 0 and 1.

        threshold_strategy : Literal['f1', 'roc'] | None
            Default: None. The strategy for determining the threshold for binary
            classification. If None, the threshold is set to 0.5.

        Returns
        -------
        LogitReport | MNLogitReport
            The appropriate regression report object is returned.
        """
        if formula is None and target is None:
            raise ValueError("target must be specified if formula is None.")

        elif formula is None:
            if predictors is None:
                predictors = self._datahandler.vars()
                if target in predictors:
                    if self._verbose:
                        print_wrapped(
                            f"Removing target variable "
                            f"{quote_and_color(target, 'yellow')} "
                            + "from predictors.",
                            type="WARNING",
                        )
                    predictors.remove(target)
            # decide between binary and multinomial logit
            df_all = self._datahandler.df_all()
            if len(df_all[target].unique()) == 2:
                return LogitReport(
                    LogitLinearModel(
                        alpha=alpha,
                        l1_weight=l1_weight,
                        threshold_strategy=threshold_strategy,
                    ),
                    self._datahandler,
                    target,
                    predictors,
                )
            else:
                return MNLogitReport(
                    MNLogitLinearModel(
                        alpha=alpha,
                        l1_weight=l1_weight,
                        threshold_strategy=threshold_strategy,
                    ),
                    self._datahandler,
                    target,
                    predictors,
                )

        else:
            try:
                y_series_train, _, X_df_train = parse_and_transform_rlike(
                    formula, self._datahandler.df_train()
                )
                y_series_test, _, X_df_test = parse_and_transform_rlike(
                    formula, self._datahandler.df_test()
                )
            except Exception as e:
                raise ValueError(f"Invalid formula: {formula}. Error: {e}.")

            # ensure missing values are dropped
            y_X_df_combined_train = pd.DataFrame(y_series_train).join(X_df_train)
            y_X_df_combined_test = pd.DataFrame(y_series_test).join(X_df_test)
            y_X_df_combined_train = y_X_df_combined_train.dropna()
            y_X_df_combined_test = y_X_df_combined_test.dropna()
            (
                y_X_df_combined_train,
                y_X_df_combined_test,
            ) = self._datahandler._force_train_test_var_agreement(
                y_X_df_combined_train, y_X_df_combined_test
            )

            predictors = y_X_df_combined_train.columns.to_list()
            target = y_series_train.name
            predictors.remove(target)

            datahandler = DataHandler(
                y_X_df_combined_train, y_X_df_combined_test, verbose=False
            )

            # decide between binary and multinomial logit
            df_all = datahandler.df_all()
            if len(df_all[target].unique()) == 2:
                return LogitReport(
                    LogitLinearModel(
                        alpha=alpha,
                        l1_weight=l1_weight,
                        threshold_strategy=threshold_strategy,
                    ),
                    datahandler,
                    target,
                    predictors,
                )
            else:
                return MNLogitReport(
                    MNLogitLinearModel(
                        alpha=alpha,
                        l1_weight=l1_weight,
                        threshold_strategy=threshold_strategy,
                    ),
                    datahandler,
                    target,
                    predictors,
                )

    # --------------------------------------------------------------------------
    # MACHINE LEARNING
    # --------------------------------------------------------------------------

    @ensure_arg_list_uniqueness()
    def regress(
        self,
        models: list[BaseR],
        target: str,
        predictors: list[str] | None = None,
        feature_selectors: list[BaseFSR] | None = None,
        max_n_features: int | None = None,
        outer_cv: int | None = None,
        outer_cv_seed: int = 42,
    ) -> MLRegressionReport:
        """Conducts a comprehensive regression ML model benchmarking exercise.
        Observations with missing data will be dropped.

        Parameters
        ----------
        models : list[BaseR]
            Models to be evaluated.

        target : str
            The variable to be predicted.

        predictors : list[str]
            Default: None.
            If None, uses all variables except target as predictors.

        feature_selectors : list[BaseFSR]
            The feature selectors for voting selection. Feature selectors
            can be used to select the most important predictors.
            Feature selectors can also be specified at the model level. If
            specified here, the same feature selectors will be used for all
            models.

        max_n_features : int
            Default: None. Maximum number of predictors to utilize.
            Ignored if feature_selectors is None.
            If None, then all features with at least 50% support are selected.

        outer_cv : int
            Default: None. If not None, reports training scores via nested k-fold CV.

        outer_cv_seed : int
            Default: 42. The random seed for the outer cross validation loop.

        Returns
        -------
        MLRegressionReport
        """
        if predictors is None:
            predictors = self._datahandler.vars()
            if target in predictors:
                predictors.remove(target)

        return MLRegressionReport(
            models=models,
            datahandler=self._datahandler,
            target=target,
            predictors=predictors,
            feature_selectors=feature_selectors,
            max_n_features=max_n_features,
            outer_cv=outer_cv,
            outer_cv_seed=outer_cv_seed,
            verbose=self._verbose,
        )

    @ensure_arg_list_uniqueness()
    def classify(
        self,
        models: list[BaseC],
        target: str,
        predictors: list[str] | None = None,
        feature_selectors: list[BaseFSC] | None = None,
        max_n_features: int | None = None,
        outer_cv: int | None = None,
        outer_cv_seed: int = 42,
    ) -> MLClassificationReport:
        """Conducts a comprehensive classification ML model benchmarking exercise.
        Observations with missing data will be dropped.

        Parameters
        ----------
        models : list[BaseC]
            Models to be evaluated.

        target : str
            The variable to be predicted.

        predictors : list[str]
            Default: None.
            If None, uses all variables except target as predictors.

        feature_selectors : list[BaseFSR]
            The feature selectors for voting selection. Feature selectors
            can be used to select the most important predictors.
            Feature selectors can also be specified at the model level. If
            specified here, the same feature selectors will be used for all
            models.

        max_n_features : int
            Default: None.
            Maximum number of predictors to utilize.
            Ignored if feature_selectors is None.
            If None, then all features with at least 50% support are selected.

        outer_cv : int
            Default: None.
            If not None, reports training scores via nested k-fold CV.

        outer_cv_seed : int
            Default: 42.
            The random seed for the outer cross validation loop.

        Returns
        -------
        MLClassificationReport
        """
        if predictors is None:
            predictors = self._datahandler.vars()
            if target in predictors:
                predictors.remove(target)

        return MLClassificationReport(
            models=models,
            datahandler=self._datahandler,
            target=target,
            predictors=predictors,
            feature_selectors=feature_selectors,
            max_n_features=max_n_features,
            outer_cv=outer_cv,
            outer_cv_seed=outer_cv_seed,
            verbose=self._verbose,
        )

    def cluster(
        self,
        models: list[BaseClust],
        features: list[str] | None = None,
        dataset: Literal["train", "all"] = "all",
    ) -> ClusterReport:
        """Conducts a clustering exercise.

        Parameters
        ----------
        models : list[BaseClust]
            Models to be evaluated.

        features : list[str] | None
            Default: None. The features to cluster on.
            If None, uses all the variables.

        dataset : Literal['train', 'all']
            Dataset to fit models on. If "train", only fits models on training data.
            Then, cluster predictions can be made on test data.
            If "all", fits models on all data.
            By default, fits models on all data.
        """
        if features is None:
            features = self._datahandler.vars()

        return ClusterReport(
            models=models,
            datahandler=self._datahandler,
            features=features,
            dataset=dataset,
        )

    # --------------------------------------------------------------------------
    # DATAHANDLER METHODS
    # --------------------------------------------------------------------------
    def load_data_checkpoint(self, checkpoint_name: str | None = None) -> "Analyzer":
        """Loads the original train and test DataFrames.

        Parameters
        ----------
        checkpoint_name : str
            Default: None. The name of the checkpoint to load.
            If None, loads the original train and test DataFrames.

        Returns
        -------
        Analyzer
            Returns self for method chaining.
        """
        self._datahandler.load_data_checkpoint(checkpoint_name)
        return self

    def save_data_checkpoint(self, checkpoint_name: str) -> "Analyzer":
        """Saves the current train and test DataFrames.

        Parameters
        ----------
        checkpoint_name : str
            The name of the checkpoint.

        Returns
        -------
        Analyzer
            Returns self for method chaining.
        """
        self._datahandler.save_data_checkpoint(checkpoint_name)
        return self

    def remove_data_checkpoint(self, checkpoint_name: str) -> "Analyzer":
        """Deletes a saved checkpoint.

        Parameters
        ----------
        checkpoint_name : str
            The name of the checkpoint to delete.

        Returns
        -------
        Analyzer
            Returns self for method chaining.
        """
        self._datahandler.remove_data_checkpoint(checkpoint_name)
        return self

    def engineer_numeric_var(self, name: str, formula: str) -> "Analyzer":
        """Engineers a new variable/feature based on a formula. The formula
        can only involve numeric variables. Creates another numeric variable.

        Parameters
        ----------
        name : str
            The name of the new variable engineered.

        formula : str
            Formula for the new feature. For example, "x1 + x2" would create
            a new feature that is the sum of the columns x1 and x2 in the DataFrame.
            All variables used must be numeric.
            Handles the following operations:
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
        >>> analyzer.engineer_numeric_feature("x3", "x1 + x2")
        >>> assert "x3" in analyzer.datahandler.vars()
        True
        >>> assert analyzer.datahandler.df_train()["x3"].equals(
        ...     analyzer.datahandler.df_train()["x1"] + analyzer.datahandler.df_train()["x2"]
        ... )
        True

        Returns
        -------
        Analyzer
            Returns self for method chaining.
        """
        self._datahandler.engineer_numeric_feature(name, formula)
        return self

    def engineer_categorical_var(
        self,
        name: str,
        numeric_var: str,
        level_names: list[str],
        thresholds: list[float],
        leq: bool = False,
    ) -> "Analyzer":
        """Engineers a new categorical variable/feature based on a list of thresholds.

        Parameters
        ----------
        name : str
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
        self._datahandler.engineer_categorical_feature(
            name, numeric_var, level_names, thresholds, leq
        )
        return self

    @ensure_arg_list_uniqueness()
    def scale(
        self,
        include_vars: list[str] | None = None,
        exclude_vars: list[str] | None = None,
        strategy: Literal["standardize", "minmax", "log", "log1p"] = "standardize",
    ) -> "Analyzer":
        """Scales the variables.

        Parameters
        ----------
        include_vars : list[str]
            Default: None. List of variables to scale.
            If None, scales values in all columns.

        exclude_vars : list[str]
            Default: None. List of variables to exclude from scaling.
            If None, no variables are excluded.

        strategy : Literal["standardize", "minmax", "log", "log1p"]
            Default: 'standardize'. The scaling strategy.

        Returns
        -------
        Analyzer
            Returns self for method chaining.
        """
        self._datahandler.scale(
            include_vars=include_vars,
            exclude_vars=exclude_vars,
            strategy=strategy,
        )
        return self

    @ensure_arg_list_uniqueness()
    def impute(
        self,
        include_vars: list[str] | None = None,
        exclude_vars: list[str] | None = None,
        numeric_strategy: Literal["median", "mean", "5nn", "10nn"] = "median",
        categorical_strategy: Literal["most_frequent", "missing"] = "most_frequent",
    ) -> "Analyzer":
        """Imputes missing values. The imputer is fit on the train DataFrame
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

        categorical_strategy : Literal['most_frequent']
            Default: 'most_frequent'.
            Strategy for imputing missing values in categorical variables.
            - 'most_frequent': impute with most frequent value.
            - 'missing': impute with 'missing'.

        Returns
        -------
        Analyzer
            Returns self for method chaining.
        """
        self._datahandler.impute(
            include_vars=include_vars,
            exclude_vars=exclude_vars,
            numeric_strategy=numeric_strategy,
            categorical_strategy=categorical_strategy,
        )
        return self

    @ensure_arg_list_uniqueness()
    def dropna(
        self,
        include_vars: list[str] | None = None,
        exclude_vars: list[str] | None = None,
    ) -> "Analyzer":
        """Drops observations (rows) with missing values on both the train
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
        Analyzer
            Returns self for method chaining.
        """
        self._datahandler.dropna(
            include_vars=include_vars,
            exclude_vars=exclude_vars,
        )
        return self

    @ensure_arg_list_uniqueness()
    def drop_highly_missing_vars(
        self,
        include_vars: list[str] | None = None,
        exclude_vars: list[str] | None = None,
        threshold: float = 0.5,
    ) -> "Analyzer":
        """Drops variables (columns) with missingness rate above a specified threshold.

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
        Analyzer
            Returns self for method chaining.
        """
        self._datahandler.drop_highly_missing_vars(
            include_vars, exclude_vars, threshold
        )
        return self

    @ensure_arg_list_uniqueness()
    def onehot(
        self,
        include_vars: list[str] | None = None,
        exclude_vars: list[str] | None = None,
        dropfirst: bool = True,
        keep_original: bool = False,
    ) -> "Analyzer":
        """One-hot encodes the specified variables (columns).

        Parameters
        ----------
        include_vars : list[str]
            Default: None. List of variables to one-hot encode.
            If None, one-hot encodes all categorical variables.

        exclude_vars : list[str]
            Default: None. List of variables to exclude from one-hot encoding.
            If None, no variables are excluded.

        dropfirst : bool
            Default: True. If True, drops the first one-hot encoded column.

        keep_original : bool
            Default: False. If True, keeps the original variables in the DataFrame.

        Returns
        -------
        Analyzer
            Returns self for method chaining.
        """
        self._datahandler.onehot(
            include_vars=include_vars,
            exclude_vars=exclude_vars,
            dropfirst=dropfirst,
            keep_original=keep_original,
        )
        return self

    @ensure_arg_list_uniqueness()
    def select_vars(
        self,
        include_vars: list[str] | None = None,
        exclude_vars: list[str] | None = None,
    ) -> "Analyzer":
        """Selects the specified variables.

        Parameters
        ----------
        include_vars : list[str]
            Default: None. List of variables to include.
            If None, includes all variables.

        exclude_vars : list[str]
            Default: None. List of variables to exclude.
            If None, no variables are excluded.

        Returns
        -------
        Analyzer
            Returns self for method chaining.
        """
        if include_vars is None:
            include_vars = sorted(self._datahandler.vars())
        if exclude_vars is not None:
            include_vars = sorted(list(set(include_vars) - set(exclude_vars)))
        self._datahandler.select_vars(vars=include_vars)
        return self

    @ensure_arg_list_uniqueness()
    def force_numeric(self, vars: list[str]) -> "Analyzer":
        """Forces specificed variables to numeric (float).

        Parameters
        ----------
        vars : list[str]
            Name of variables to force to numeric.

        Returns
        -------
        Analyzer
            Returns self for method chaining.
        """
        self._datahandler.force_numeric(vars)
        return self

    @ensure_arg_list_uniqueness()
    def force_categorical(self, vars: list[str]) -> "Analyzer":
        """Forces specificed variables (columns) to have categorical values.
        That is, the variables' values are converted to strings.

        Parameters
        ----------
        vars : list[str]
            Name of variables to force to categorical.

        Returns
        -------
        Analyzer
            Returns self for method chaining.
        """
        self._datahandler.force_categorical(vars)
        return self

    @ensure_arg_list_uniqueness()
    def force_binary(
        self,
        var: str,
        pos_label: str | None = None,
        ignore_multiclass: bool = True,
        rename: bool = True,
    ) -> "Analyzer":
        """Forces variables to be binary (0 and 1 valued numeric variables).
        Does nothing if the data contains more than two classes unless
        ignore_multiclass is True and pos_label is specified,
        in which case all classes except pos_label are labeled with zero.

        Parameters
        ----------
        vars : str
            Name of variable to force to binary.

        pos_labels : str
            Default: None. The positive label.
            If None, the most common class is labeled as the positive class.

        ignore_multiclass : bool
            Default: False. If True, all classes except pos_label are labeled with
            zero. Otherwise raises ValueError.

        rename : bool
            Default: True. If True, the variable is renamed to
            {var}::{pos_label}.

        Returns
        -------
        Analyzer
            Returns self for method chaining.
        """
        if pos_label is None:
            # set the positive label to the most common class
            pos_label = self._datahandler.df_all()[var].value_counts().index[0]
        self._datahandler.force_binary(
            vars=[var],
            pos_labels=[pos_label],
            ignore_multiclass=ignore_multiclass,
            rename=rename,
        )
        return self

    def datahandler(self) -> DataHandler:
        """Returns the DataHandler.

        Returns
        -------
        DataHandler
            The DataHandler object takes care of data management and preprocessing.
        """
        return self._datahandler

    def numeric_vars(self) -> list[str]:
        """Returns the numeric variables in the working train DataFrame.

        Returns
        -------
        list[str]
            The numeric variables.
        """
        return self._datahandler.numeric_vars()

    def categorical_vars(self) -> list[str]:
        """Returns the categorical variables in the working train DataFrame.

        Returns
        -------
        list[str]
            The categorical variables.
        """
        return self._datahandler.categorical_vars()

    def vars(self) -> list[str]:
        """Returns the variables in the working train DataFrame.

        Returns
        -------
        list[str]
            The variables.
        """
        return self._datahandler.vars()

    def shape(self, dataset: Literal["train", "test"]) -> tuple[int, int]:
        """Returns the shape of the working train DataFrame.

        Parameters
        ----------
        dataset : Literal['train', 'test']
            The dataset to get the shape of.

        Returns
        -------
        tuple[int, int]
            The shape of the working DataFrame.
        """
        if dataset == "train":
            return self._datahandler._working_df_train.shape
        elif dataset == "test":
            return self._datahandler._working_df_test.shape
        else:
            raise ValueError(f"Invalid input: dataset = {dataset}.")

    def value_counts(
        self, var: str, dataset: Literal["train", "test", "both"] = "both"
    ) -> pd.Series:
        """Returns the value counts of a variable in the working train DataFrame.

        Parameters
        ----------
        var : str
            The variable to get the value counts of.

        dataset : Literal['train', 'test']
            The dataset to get the value counts of.

        Returns
        -------
        pd.Series
            The value counts of the variable.
        """
        if dataset == "train":
            return self._datahandler._working_df_train[var].value_counts()
        elif dataset == "test":
            return self._datahandler._working_df_test[var].value_counts()
        elif dataset == "both":
            return self._datahandler.df_all()[var].value_counts()
        else:
            raise ValueError(f"Invalid input: dataset = {dataset}.")

    def df_train(self) -> pd.DataFrame:
        """Returns the working train DataFrame.

        Returns
        -------
        pd.DataFrame
            The working train DataFrame.
        """
        return self._datahandler.df_train()

    def df_test(self) -> pd.DataFrame:
        """Returns the working test DataFrame.

        Returns
        -------
        pd.DataFrame
            The working test DataFrame.
        """
        return self._datahandler.df_test()

    def df_all(self) -> pd.DataFrame:
        """Returns the working DataFrame.

        Returns
        -------
        pd.DataFrame
            The working DataFrame.
        """
        return self._datahandler.df_all()

    # --------------------------------------------------------------------------
    # MAGIC METHODS
    # --------------------------------------------------------------------------
    def __len__(self) -> int:
        """Returns the number of units (rows) in working train DataFrame."""
        return len(self._datahandler)

    def __str__(self) -> str:
        """Returns metadata in string form."""
        return self._datahandler.__str__()

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))
