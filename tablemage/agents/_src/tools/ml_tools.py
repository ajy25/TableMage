from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field
from functools import partial
from typing import Literal
from .tooling_context import ToolingContext
from .tooling_utils import tooling_decorator
from .._debug.logger import print_debug
from ....ml import (
    LinearC,
    LinearR,
    SVMC,
    SVMR,
    TreesC,
    TreesR,
    MLPC,
    MLPR,
    GMMClust,
    KMeansClust,
)
from ...._base import BaseC, BaseR
from ....fs import (
    BorutaFSC,
    BorutaFSR,
    KBestFSC,
    KBestFSR,
)
from ..._src.options import options


def parse_model_list_from_str(
    models_str: str,
    type: Literal["classification", "regression"],
    n_jobs: int = options._cpu_count,
) -> tuple[list[BaseC] | list[BaseR], list[str]]:
    list_of_models = [model_str.strip() for model_str in models_str.split(",")]
    output = []
    output_code = []
    if type == "regression":
        for model_str in list_of_models:
            if model_str == "OLS":
                output.append(LinearR("ols", name=model_str))
                output_code.append("LinearR('ols', name='OLS')")
            elif model_str == "Ridge":
                output.append(LinearR("l2", name=model_str, n_jobs=n_jobs))
                output_code.append(f"LinearR('l2', name='Ridge', n_jobs={n_jobs})")
            elif model_str == "Lasso":
                output.append(LinearR("l1", name=model_str, n_jobs=n_jobs))
                output_code.append(f"LinearR('l1', name='Lasso', n_jobs={n_jobs})")
            elif model_str == "ElasticNet":
                output.append(LinearR("elasticnet", name=model_str, n_jobs=n_jobs))
                output_code.append(
                    f"LinearR('elasticnet', name='ElasticNet', n_jobs={n_jobs})"
                )
            elif model_str == "RF":
                output.append(TreesR("random_forest", name=model_str, n_jobs=n_jobs))
                output_code.append(
                    f"TreesR('random_forest', name='RF', n_jobs={n_jobs})"
                )
            elif model_str == "XGBoost":
                output.append(TreesR("xgboost", name=model_str, n_jobs=n_jobs))
                output_code.append(
                    f"TreesR('xgboost', name='XGBoost', n_jobs={n_jobs})"
                )
            elif model_str == "SVM":
                output.append(SVMR("rbf", name=model_str, n_jobs=n_jobs))
                output_code.append(f"SVMR('rbf', name='SVM', n_jobs={n_jobs})")
            elif model_str == "MLP":
                output.append(MLPR(name=model_str, n_jobs=n_jobs))
                output_code.append(f"MLPR(name='MLP', n_jobs={n_jobs})")
            else:
                raise ValueError(f"Invalid model specification: {model_str}")
    elif type == "classification":
        for model_str in list_of_models:
            if model_str == "Logistic":
                output.append(LinearC("no_penalty", name=model_str))
                output_code.append("LinearC('no_penalty', name='Logistic'")
            elif model_str == "Ridge":
                output.append(LinearC("l2", name=model_str, n_jobs=n_jobs))
                output_code.append(f"LinearC('l2', name='Ridge', n_jobs={n_jobs})")
            elif model_str == "Lasso":
                output.append(LinearC("l1", name=model_str, n_jobs=n_jobs))
                output_code.append(f"LinearC('l1', name='Lasso', n_jobs={n_jobs})")
            elif model_str == "ElasticNet":
                output.append(LinearC("elasticnet", name=model_str, n_jobs=n_jobs))
                output_code.append(
                    f"LinearC('elasticnet', name='ElasticNet', n_jobs={n_jobs})"
                )
            elif model_str == "RF":
                output.append(TreesC("random_forest", name=model_str, n_jobs=n_jobs))
                output_code.append(
                    f"TreesC('random_forest', name='RF', n_jobs={n_jobs})"
                )
            elif model_str == "XGBoost":
                output.append(TreesC("xgboost", name=model_str, n_jobs=n_jobs))
                output_code.append(
                    f"TreesC('xgboost', name='XGBoost', n_jobs={n_jobs})"
                )
            elif model_str == "SVM":
                output.append(SVMC("rbf", name=model_str, n_jobs=n_jobs))
                output_code.append(f"SVMC('rbf', name='SVM', n_jobs={n_jobs})")
            elif model_str == "MLP":
                output.append(MLPC(name=model_str, n_jobs=n_jobs))
                output_code.append(f"MLPC(name='MLP', n_jobs={n_jobs})")
            else:
                raise ValueError(f"Invalid model specification: {model_str}")
    else:
        raise ValueError(f"Invalid ML problem type: {type}")

    return output, output_code


@tooling_decorator
def parse_predictor_list_from_str(predictors_str: str) -> list[str]:
    return [predictor.strip() for predictor in predictors_str.split(",")]


class _MLRegressionInput(BaseModel):
    models: str = Field(
        description="""\
A comma delimited string of machine learning models to evaluate.
The available models are (in 'Model Name': Description format)...
1. 'OLS': Ordinary least squares regression
2. 'Ridge': Linear regression with L2 penalty
3. 'Lasso': Linear regression with L1 penalty
4. 'ElasticNet': Linear regression with L1 and L2 penalty
5. 'RF': Random forest regressor
6. 'XGBoost': XGBoost regressor
7. 'SVM': Support vector machine regressor with radial basis function kernel
An example input (without the quotes) is: 'OLS, Lasso, RF'.
"""
    )
    target: str = Field(description="The target variable.")
    predictors: str = Field(
        description="A comma delimited string of features/predictors. "
        "An example input (without the quotes) is: 'var1, var2, var3'."
    )


@tooling_decorator
def _ml_regression_function(
    models: str, target: str, predictors: str, context: ToolingContext
) -> str:
    models_list, models_list_code = parse_model_list_from_str(models, type="regression")
    models_list_str = ", ".join([model._name for model in models_list])
    print_debug("_ml_regression_function called")
    print_debug("_ml_regression_function Models to test: " + models_list_str)
    predictors_list = parse_predictor_list_from_str(predictors)
    print_debug("_ml_regression_function Target: " + target)
    print_debug("_ml_regression_function Predictors: " + str(predictors_list))

    context.add_thought(
        "I am going to predict {target} with {predictors} using models {models}.".format(
            target=target, predictors=", ".join(predictors_list), models=models_list_str
        )
        + " This might take a while."
    )

    models_str_code = ", ".join(models_list_code)

    context.add_code(
        "analyzer.regress(models=[{models_str_code}], target='{target}', predictors=['{predictors}'])".format(
            models_str_code=models_str_code,
            target=target,
            predictors="', '".join(predictors_list),
        )
    )

    report = context._data_container.analyzer.regress(
        models=models_list, target=target, predictors=predictors_list
    )

    context.add_table(table=report.metrics("both"), add_to_vectorstore=False)
    output_str = context.add_dict(report._to_dict())
    return output_str


def build_ml_regression_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_ml_regression_function, context=context),
        name="ml_regression_function",
        description="Performs regression with a list of machine learning models that you must specify. "
        "Predicts the target variable with a list of predictor variables. "
        "Model hyperparameters are optimized automatically using cross-validation and the Optuna library. "
        "Returns a string describing model performances, feature importances, and hyperparameters (searched and best).",
        fn_schema=_MLRegressionInput,
    )


class _MLClassificationInput(BaseModel):
    models: str = Field(
        description="""\
A comma delimited string of machine learning models to evaluate.
The available models are (in 'Model Name': Description format)...
1. 'Logistic': Logistic regression
2. 'Ridge': Logistic regression with L2 penalty
3. 'Lasso': Logistic regression with L1 penalty
4. 'ElasticNet': Logistic regression with L1 and L2 penalty
5. 'RF': Random forest classifier
6. 'XGBoost': XGBoost classifier
7. 'SVM': Support vector machine classifier with radial basis function kernel
An example input (without the quotes) is: 'Logistic, RF, XGBoost'.
"""
    )
    target: str = Field(description="The target variable.")
    predictors: str = Field(
        description="A comma delimited string of features/predictors. "
        "An example input (without the quotes) is: 'var1, var2, var3'."
    )


@tooling_decorator
def _ml_classification_function(
    models: str, target: str, predictors: str, context: ToolingContext
) -> str:
    models_list, models_list_code = parse_model_list_from_str(
        models, type="classification"
    )
    models_list_str = ", ".join([model._name for model in models_list])

    print_debug("_ml_classification_function called")
    print_debug("_ml_classification_function Models to test: " + models_list_str)

    predictors_list = parse_predictor_list_from_str(predictors)

    print_debug("_ml_classification_function Target: " + target)
    print_debug("_ml_classification_function Predictors: " + str(predictors_list))

    context.add_thought(
        "I am going to predict {target} with {predictors} using models {models}.".format(
            target=target, predictors=", ".join(predictors_list), models=models_list_str
        )
        + " This might take a while."
    )
    models_str_code = ", ".join(models_list_code)
    context.add_code(
        "analyzer.classify(models=[{models_str_code}], target='{target}', predictors=['{predictors}'])".format(
            models_str_code=models_str_code,
            target=target,
            predictors="', '".join(predictors_list),
        )
    )

    report = context._data_container.analyzer.classify(
        models=models_list, target=target, predictors=predictors_list
    )

    context.add_table(table=report.metrics("both"), add_to_vectorstore=False)
    output_str = context.add_dict(report._to_dict())
    return output_str


def build_ml_classification_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_ml_classification_function, context=context),
        name="ml_classification_function",
        description="Performs classification with a list of machine learning models that you must specify. "
        "Predicts the target variable with a list of predictor variables. "
        "Model hyperparameters are optimized automatically using cross-validation and the Optuna library. "
        "Returns a string describing model performances, feature importances, and hyperparameters (searched and best).",
        fn_schema=_MLClassificationInput,
    )


class _FeatureSelectionInput(BaseModel):
    feature_selector: str = Field(
        description="""\
The feature selection method to use. The available methods are:

1. 'Boruta': Boruta method (automatically selects the number of features)
2. 'Select<N>Best': Select N best features based on the F-score, where you replace <N> with the number of features you want to select.

Two example inputs (without the quotes) are: 'Boruta' and 'Select5Best'.
"""
    )
    target: str = Field(description="The target variable.")
    predictors: str = Field(
        description="""\
A comma delimited string of variables used by the models to predict the target.
An example input (without the quotes) is: 'var1, var2, var3'.
"""
    )
    task: Literal["classification", "regression"] = Field(
        description="The type of task to perform. "
        "Either 'classification' or 'regression'."
    )


@tooling_decorator
def _feature_selection_function(
    feature_selector: str,
    target: str,
    predictors: str,
    task: Literal["classification", "regression"],
    context: ToolingContext,
) -> str:
    print_debug("_feature_selection_function called")
    print_debug("_feature_selection_function Feature selector: " + feature_selector)
    predictors_list = parse_predictor_list_from_str(predictors)
    print_debug("_feature_selection_function Target: " + target)
    print_debug("_feature_selection_function Predictors: " + str(predictors_list))

    # figure out if target is categorical or continuous
    if target in context._data_container.analyzer.numeric_vars():
        if task == "classification":
            target_type = "classification"
        elif task == "regression":
            target_type = "regression"
        else:
            raise ValueError(f"Invalid task type: {task}.")
    elif target in context._data_container.analyzer.categorical_vars():
        if task == "regression":
            raise ValueError(
                f"Target variable {target} is categorical, but task is regression."
            )
        target_type = "classification"
    else:
        raise ValueError(f"Target variable {target} is not found in the dataset.")

    if feature_selector == "Boruta":
        fs = BorutaFSC() if target_type == "classification" else BorutaFSR()
        fs_code = "BorutaFSC()" if target_type == "classification" else "BorutaFSR()"
    else:
        # should be of the form "Select{N}Best"
        if not feature_selector.startswith("Select") or not feature_selector.endswith(
            "Best"
        ):
            raise ValueError(
                "Invalid feature selector. "
                "Should be either 'Boruta' or 'Select<N>Best'."
            )
        k = int(feature_selector[6:-4])
        fs = (
            KBestFSC(k=k, scorer="f_classif")
            if target_type == "classification"
            else KBestFSR(k=k, scorer="f_regression")
        )
        fs_code = (
            f"KBestFSC(k={k}, scorer='f_classif')"
            if target_type == "classification"
            else f"KBestFSR(k={k}, scorer='f_regression')"
        )

    context.add_thought(
        "I am going to select features to predict {target} with {predictors} using the {feature_selector} method.".format(
            target=target,
            predictors=", ".join(predictors_list),
            feature_selector=feature_selector,
        )
    )

    context.add_code(
        "analyzer.select_features(feature_selectors=[{fs_code}], target='{target}', predictors=['{predictors}'])".format(
            fs_code=fs_code,
            target=target,
            predictors="', '".join(predictors_list),
        )
    )

    report = context._data_container.analyzer.select_features(
        feature_selectors=[fs], target=target, predictors=predictors_list
    )

    output_dict = report._to_dict()
    context.add_dict(output_dict)
    output_str = f"Selected features: {', '.join(report.top_features())}. "
    output_str += f"Consider using only these features to predict {target}."
    return output_str


def build_feature_selection_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_feature_selection_function, context=context),
        name="feature_selection_function",
        description="""\
Performs feature selection with a specified method. \
Automatically detecs if the target variable is categorical or continuous. \
Selects the best features/predictors/variables from a list of predictor variables to predict the target variable. \
Returns a string describing the selected features. \
Categorical variables are one-hot encoded before feature selection. \
If a category is selected, the output would be '<variable_name>::<category>'.
""",
        fn_schema=_FeatureSelectionInput,
    )


class _ClusteringInput(BaseModel):
    features: str = Field(
        description="""\
A comma delimited string of variables/features to use for clustering.
An example input (without the quotes) is: 'var1, var2, var3'.
"""
    )
    model: str = Field(
        description="""\
The available models are (in 'Model Name': Description format):

1. 'KMeans': KMeans clustering
2. 'GMM': Gaussian mixture model clustering

An example input (without the quotes) is: 'KMeans'.
"""
    )
    n_clusters: int = Field(
        description="The number of clusters to create. An example input (without the quotes) is: '5'. "
        "If left blank (empty str), the optimal number of clusters will be automatically determined."
    )
    max_n_clusters: int = Field(
        description="The maximum number of clusters to test. An example input (without the quotes) is: '10'. "
        "This is only used if 'n_clusters' is left blank (empty str). Leave blank if 'n_clusters' is specified."
    )
    vis_type: str = Field(
        description="""\
The type of visualization to use. The available types are:

1. 'PCA': Principal component analysis
2. 'TSNE': t-distributed stochastic neighbor embedding

An example input (without the quotes) is: 'PCA'.
"""
    )


@tooling_decorator
def _clustering_function(
    features: str,
    model: str,
    n_clusters: int,
    max_n_clusters: int,
    vis_type: str,
    context: ToolingContext,
) -> str:
    """Clustering function."""

    n_clusters = int(n_clusters) if n_clusters != "" else None
    max_n_clusters = int(max_n_clusters) if max_n_clusters != "" else 10

    print_debug("_clustering_function called")
    print_debug("_clustering_function Features: " + features)
    print_debug("_clustering_function Model: " + model)
    print_debug("_clustering_function Number of clusters: " + str(n_clusters))
    print_debug("_clustering_function Max number of clusters: " + str(max_n_clusters))

    features_list = parse_predictor_list_from_str(features)
    print_debug("_clustering_function Features: " + str(features_list))

    if model == "KMeans":
        clust = KMeansClust(k=n_clusters, max_k=max_n_clusters)
        clust_code = f"KMeansClust(k={n_clusters}, max_k={max_n_clusters})"
    elif model == "GMM":
        clust = GMMClust(n_components=n_clusters, max_n_components=max_n_clusters)
        clust_code = (
            f"GMMClust(n_components={n_clusters}, max_n_components={max_n_clusters})"
        )
    else:
        raise ValueError(f"Invalid clustering model: {model}")

    context.add_thought(
        "I am going to cluster the data using the {model} model with {n_clusters} clusters.".format(
            model=model, n_clusters=n_clusters
        )
    )

    context.add_code(
        "analyzer.cluster(models=[{clust_code}], features=['{features}'])".format(
            clust_code=clust_code, features="', '".join(features_list)
        )
    )

    report = context._data_container.analyzer.cluster(
        models=[clust], features=features_list
    )

    output_str = context.add_figure(
        fig=report.plot_clusters_2d(
            model_id=clust._name,
            dim_reduction_method="pca" if vis_type == "PCA" else "tsne",
        ),
        text_description="{vistype} clustering visualization of the data, considering the features {features}.".format(
            vistype=vis_type, features=", ".join(features_list)
        ),
    )

    return output_str


def build_clustering_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_clustering_function, context=context),
        name="clustering_function",
        description="""Performs clustering with a specified method. \
Clusters the data using a list of variables. \
Returns a figure showing the clusters.
""",
        fn_schema=_ClusteringInput,
    )
