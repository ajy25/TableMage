from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field
from functools import partial
from .tooling_context import ToolingContext
from .tooling_utils import tooling_decorator


# t-test tool
class _TTestInput(BaseModel):
    numeric_var: str = Field(
        description="The numeric variable to perform the t-test on."
    )
    binary_var: str = Field(description="The binary variable to split the data on.")
    test: str = Field(
        "The type of t-test to perform. Options: `welch`, `student`, `mann-whitney`."
    )


@tooling_decorator
def _t_test_function(
    numeric_var: str,
    binary_var: str,
    test: str = "welch",
    context: ToolingContext = None,
) -> str:
    context.add_thought(
        "I am going to perform a t-test on the variable {numeric_var} split by the binary variable {binary_var}.".format(
            numeric_var=numeric_var, binary_var=binary_var
        )
    )
    context.add_code(
        "analyzer.eda().ttest(numeric_var='{}', stratify_by='{}', strategy='{}')".format(
            numeric_var, binary_var, test
        )
    )
    dict_output = (
        context._data_container.analyzer.eda("all")
        .ttest(numeric_var=numeric_var, stratify_by=binary_var, strategy=test)
        ._to_dict()
    )
    return context.add_dict(dict_output)


def build_test_ttest_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_t_test_function, context=context),
        name="t_test_function",
        description="""\
Performs a t-test on a numeric variable split by a binary variable. \
Tests the null hypothesis that the means of the groups are equal. \
Returns a JSON string containing the results and which test was used.\
""",
        fn_schema=_TTestInput,
    )


# ANOVA test tool
class _AnovaTestInput(BaseModel):
    numeric_var: str = Field(
        description="The numeric variable to perform the ANOVA test on."
    )
    categorical_var: str = Field(
        description="The categorical variable to split the data on."
    )
    test: str = Field(
        description="The type of ANOVA test to perform. Options: `anova_oneway`, `kruskal`."
    )


@tooling_decorator
def _anova_test_function(
    numeric_var: str,
    categorical_var: str,
    test: str = "anova_oneway",
    context: ToolingContext = None,
) -> str:
    context.add_thought(
        "I am going to perform an ANOVA test on the variable {numeric_var} split by the categorical variable {categorical_var}.".format(
            numeric_var=numeric_var, categorical_var=categorical_var
        )
    )
    context.add_code(
        "analyzer.eda().anova(numeric_var='{}', stratify_by='{}', strategy='{}')".format(
            numeric_var, categorical_var, test
        )
    )
    dict_output = (
        context._data_container.analyzer.eda("all")
        .anova(numeric_var=numeric_var, stratify_by=categorical_var, strategy=test)
        ._to_dict()
    )
    return context.add_dict(dict_output)


def build_test_anova_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_anova_test_function, context=context),
        name="anova_test_function",
        description="""\
Performs an ANOVA test on a numeric variable split by a categorical variable. \
Tests the null hypothesis that the means of the groups are equal. \
Returns a JSON string containing the results and which test was used.\
""",
        fn_schema=_AnovaTestInput,
    )


# Normality test tool
class _TestNormalityInput(BaseModel):
    numeric_var: str = Field(description="The numeric variable to test for normality.")
    test: str = Field("The test to perform. Options: `shapiro`, `kstest`, `anderson`.")


@tooling_decorator
def _test_normality_function(
    numeric_var: str,
    test: str = "shapiro",
    context: ToolingContext = None,
) -> str:
    context.add_thought(
        "I am going to test whether the variable {numeric_var} is normally distributed.".format(
            numeric_var=numeric_var
        )
    )
    context.add_code(
        "analyzer.eda().test_normality(numeric_var='{}', method='{}')".format(
            numeric_var, test
        )
    )
    dict_output = (
        context._data_container.analyzer.eda("all")
        .test_normality(numeric_var=numeric_var, method=test)
        ._to_dict()
    )
    return context.add_dict(dict_output)


def build_test_normality_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_test_normality_function, context=context),
        name="test_normality_function",
        description="""Tests whether a numeric variable is normally distributed. \
The null hypothesis is that the data is normally distributed.
Returns a JSON string containing results and which test used.
""",
        fn_schema=_TestNormalityInput,
    )


# chi2 test tool
class _Chi2TestInput(BaseModel):
    x: str = Field(description="The first categorical variable.")
    y: str = Field(description="The second categorical variable.")


@tooling_decorator
def _chi2_test_function(x: str, y: str, context: ToolingContext = None) -> str:
    context.add_thought(
        "I am going to perform a chi-squared test of independence between the variables {x} and {y}.".format(
            x=x, y=y
        )
    )
    context.add_code(
        "analyzer.eda().chi2(categorical_var_1='{}', categorical_var_2='{}')".format(
            x, y
        )
    )
    dict_output = (
        context._data_container.analyzer.eda("all")
        .chi2(categorical_var_1=x, categorical_var_2=y)
        ._to_dict()
    )
    return context.add_dict(dict_output)


def build_test_chi2_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_chi2_test_function, context=context),
        name="chi2_test_function",
        description="""Performs a chi-squared test of independence between two categorical variables. \
The null hypothesis is that the variables are independent. \
Returns a JSON string containing results and which test used.""",
        fn_schema=_Chi2TestInput,
    )


# Plot distribution tool
class _PlotInput(BaseModel):
    x: str = Field(description="Variable to place on the x-axis of plot.")
    y: str = Field(
        description="Variable to place on the y-axis of plot. "
        "If an empty string ('') is provided, the distribution of x will be plotted. "
        "Otherwise, a scatter plot, box plot, or crosstab heatmap will be generated, "
        "depending on the data types of x and y.",
    )


@tooling_decorator
def _plot_function(x: str, y: str = "", context: ToolingContext = None) -> str:
    if y == "":
        context.add_thought(
            "I am going to plot the distribution of the variable: {x}.".format(x=x)
        )
        context.add_code("analyzer.eda().plot('{x}')".format(x=x))
        fig = context._data_container.analyzer.eda("all").plot(x)
        return context.add_figure(
            fig, text_description=f"Distribution plot of variable: {x}."
        )
    else:
        context.add_thought(
            "I am going to plot the relationship between {x} and {y}.".format(x=x, y=y)
        )
        context.add_code("analyzer.eda().plot('{x}', '{y}')".format(x=x, y=y))
        fig = context._data_container.analyzer.eda("all").plot(x, y)
        # determine plot type
        if x in context._data_container.analyzer.eda("all").categorical_vars():
            if y in context._data_container.analyzer.eda("all").categorical_vars():
                text_description = f"Crosstab heatmap of variables: {x} and {y}."
            else:
                text_description = f"Box plot of variables: {x} and {y}."
        else:
            if y in context._data_container.analyzer.eda("all").categorical_vars():
                text_description = f"Box plot of variables: {y} and {x}."
            else:
                text_description = f"Scatter plot of variables: {x} and {y}."

        return context.add_figure(fig, text_description=text_description)


def build_plot_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_plot_function, context=context),
        name="plot_function",
        description="""\
Versatile plotting tool. Provide one or two variables, categorical or numeric. \
If one variable is provided, a distribution plot will be generated \
(histogram for numeric, bar plot for categorical). \
If two variables are provided, a scatter plot, box plot, \
or crosstab heatmap will be generated, \
depending on the data types of the variables. \
Returns a JSON string describing the figure.\
""",
        fn_schema=_PlotInput,
    )


# Pairplot tool


class _PlotPairsInput(BaseModel):
    vars: str = Field(
        description="Comma delimited list of variables to include in the pairplot."
    )


@tooling_decorator
def _plot_pairs_function(
    vars: str,
    context: ToolingContext = None,
) -> str:
    vars_list = [var.strip() for var in vars.split(",")]
    context.add_thought(
        "I am going to generate a pairplot for the following variables: {vars_list}.".format(
            vars_list=vars_list
        )
    )
    context.add_code("analyzer.eda().plot_pairs(['{}'])".format("', '".join(vars_list)))
    fig = context._data_container.analyzer.eda("all").plot_pairs(vars_list)
    return context.add_figure(
        fig, text_description=f"Pairplot of variables: {', '.join(vars_list)}."
    )


def build_plot_pairs_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_plot_pairs_function, context=context),
        name="plot_pairs_function",
        description="""\
Generates a pairplot for the specified variables. \
Returns a JSON string describing the pairplot figure.\
""",
        fn_schema=_PlotPairsInput,
    )


# Correlation comparison tool
class _CorrelationComparisonInput(BaseModel):
    target: str = Field(description="The target variable to compare correlations with.")
    numeric_vars: str = Field(
        description="Comma delimited list of variables with which to compare correlations with the target variable.",
    )


@tooling_decorator
def _correlation_comparison_function(
    target: str,
    numeric_vars: str = "",
    context: ToolingContext = None,
) -> str:
    if target not in context.data_container.analyzer.eda("all").numeric_vars():
        raise ValueError("The target variable must be numeric.")
    if numeric_vars == "":
        # use all numeric variables
        vars_list = context.data_container.analyzer.eda("all").numeric_vars()
    else:
        vars_list = [var.strip() for var in numeric_vars.split(",")]
    df_output = context.data_container.analyzer.eda(
        "all"
    ).tabulate_correlation_comparison(
        target=target,
        numeric_vars=vars_list,
    )
    context.add_thought(
        "I am going to compare the correlation of {target} with the following variables: {vars_list}.".format(
            target=target, vars_list=", ".join(vars_list)
        )
    )
    context.add_code(
        "analyzer.eda().tabulate_correlation_comparison(target='{}', numeric_vars=['{}'])".format(
            target, "', '".join(vars_list)
        )
    )
    return context.add_table(df_output, add_to_vectorstore=True)


def build_correlation_comparison_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_correlation_comparison_function, context=context),
        name="correlation_comparison_function",
        description="Compares the correlation of a target variable with other numeric variables. "
        "Returns a JSON string containing the correlation values.",
        fn_schema=_CorrelationComparisonInput,
    )


# Correlation matrix tool
class _CorrelationMatrixInput(BaseModel):
    numeric_vars: str = Field(
        description="Comma delimited list of numeric variables to include in the correlation matrix. "
    )


@tooling_decorator
def _correlation_matrix_function(
    numeric_vars: str = "", context: ToolingContext = None
) -> str:
    if numeric_vars == "":
        # use all numeric variables
        numeric_vars_list = context._data_container.analyzer.eda("all").numeric_vars()
    else:
        numeric_vars_list = [var.strip() for var in numeric_vars.split(",")]

    df_output = context._data_container.analyzer.eda("all").tabulate_correlation_matrix(
        numeric_vars=numeric_vars_list
    )
    context.add_thought(
        "I am going to compute the correlation matrix for the following variables: {vars_list}.".format(
            vars_list=", ".join(numeric_vars_list)
        )
    )
    context.add_code(
        "analyzer.eda().tabulate_correlation_matrix(numeric_vars=['{}'])".format(
            "', '".join(numeric_vars_list)
        )
    )
    return context.add_table(df_output, add_to_vectorstore=True)


def build_correlation_matrix_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_correlation_matrix_function, context=context),
        name="correlation_matrix_function",
        description="Computes a correlation matrix for the specified numeric variables. "
        "Returns a JSON string containing the correlation matrix.",
        fn_schema=_CorrelationMatrixInput,
    )


class _ValueCountsInput(BaseModel):
    categorical_var: str = Field(
        description="The categorical variable to tabulate value counts for."
    )


def _value_counts_function(
    categorical_var: str,
    context: ToolingContext = None,
):
    df_output = context._data_container.analyzer.eda("all").value_counts(
        categorical_var
    )
    context.add_thought(
        "I am going to generate value counts for the variable: {var}.".format(
            var=categorical_var
        )
    )
    context.add_code("analyzer.eda().value_counts('{var}')".format(var=categorical_var))
    return context.add_table(df_output, add_to_vectorstore=True)


def build_value_counts_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_value_counts_function, context=context),
        name="value_counts_function",
        description="""\
Generates value counts for a categorical variable. \
Returns a JSON string containing the value counts.\
""",
        fn_schema=_ValueCountsInput,
    )


class _BlankInput(BaseModel):
    pass


# Numeric summary statistics tool
@tooling_decorator
def _numeric_summary_statistics_function(context: ToolingContext) -> str:
    df_output = context._data_container.analyzer.eda("all").numeric_stats()
    context.add_thought(
        "I am going to generate summary statistics for the numeric variables in the dataset."
    )
    context.add_code("analyzer.eda().numeric_stats()")
    return context.add_table(df_output, add_to_vectorstore=True)


def build_numeric_summary_statistics_tool(context: ToolingContext) -> FunctionTool:
    def temp_fn():
        return _numeric_summary_statistics_function(context)

    return FunctionTool.from_defaults(
        fn=temp_fn,
        name="numeric_summary_statistics_function",
        description="""
Generates summary statistics for the numeric variables in the dataset. \
Returns a JSON string containing the summary statistics, including \
mean, median, standard deviation, variance, minimum, maximum, and missingness.\
""",
        fn_schema=_BlankInput(),
    )


# Categorical summary statistics tool
def _categorical_summary_statistics_function(context: ToolingContext) -> str:
    """Generates categorical summary statistics for the dataset."""
    df_output = context._data_container.analyzer.eda("all").categorical_stats()
    context.add_thought(
        "I am going to generate summary statistics for the categorical variables in the dataset."
    )
    context.add_code("analyzer.eda().categorical_stats()")
    return context.add_table(df_output, add_to_vectorstore=True)


@tooling_decorator
def build_categorical_summary_statistics_tool(context: ToolingContext) -> FunctionTool:
    def temp_fn():
        return _categorical_summary_statistics_function(context)

    return FunctionTool.from_defaults(
        fn=temp_fn,
        name="categorical_summary_statistics_function",
        description="""\
Generates summary statistics for the categorical variables in the dataset. \
Returns a JSON string containing the summary statistics, \
including most frequent category, least frequent category, and missingness.\
""",
        fn_schema=_BlankInput(),
    )
