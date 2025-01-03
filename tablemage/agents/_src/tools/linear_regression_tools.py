from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field
from functools import partial
from .tooling_context import ToolingContext
from .._debug.logger import print_debug
from .tooling_utils import tooling_decorator


def parse_predictor_list_from_str(predictors_str: str) -> list[str]:
    return [predictor.strip() for predictor in predictors_str.split(",")]


# OLS tool
class _OLSToolInput(BaseModel):
    target: str = Field(
        description="The target variable, i.e. the variable to predict."
    )

    predictors: str = Field(
        description="A comma delimited string of variables used by the models to predict the target. "
        "An example input (without the quotes) is: `var1, var2, var3`."
    )


@tooling_decorator
def _ols_function(target: str, predictors: str, context: ToolingContext) -> str:
    """Performs ordinary least squares regression."""
    print_debug(f"_ols_function call: " f"predictors: {predictors}, target: {target}")
    context.add_thought(
        "I am going to perform ordinary least squares regression to predict {target} using the predictors: {predictors}.".format(
            target=target, predictors=predictors
        )
    )
    context.add_code(
        f"analyzer.ols(target='{target}', predictors={parse_predictor_list_from_str(predictors)})"
    )

    ols_report = context._data_container.analyzer.ols(
        target=target.strip(), predictors=parse_predictor_list_from_str(predictors)
    )
    print_debug(f"summary: {str(ols_report)}")
    context.add_table(ols_report.metrics("both"), add_to_vectorstore=False)
    context.add_table(ols_report.coefs(), add_to_vectorstore=False)
    output_str = context.add_dict(ols_report._to_dict())
    output_str += "\n" + context.add_figure(
        ols_report.plot_diagnostics("train"), text_description="Diagnostic plots."
    )
    print_debug(f"_ols_function output: {output_str}")
    return output_str


def build_ols_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_ols_function, context=context),
        name="ols_function",
        description="""Performs ordinary least squares regression.
        Returns a JSON string containing coefficients and metrics.
        Also, plots diagnostic plots.""",
        fn_schema=_OLSToolInput,
    )


# Logit tool
class _LogitToolInput(BaseModel):
    target: str = Field(
        description="The target variable, i.e. the variable to predict."
    )
    predictors: str = Field(
        description="A comma delimited string of variables used by the models to predict the target. "
        "An example input (without the quotes) is: `var1, var2, var3`."
    )


@tooling_decorator
def _logit_function(target: str, predictors: str, context: ToolingContext) -> str:
    """Performs logistic regression."""
    print_debug(f"_logit_function call: " f"predictors: {predictors}, target: {target}")
    context.add_thought(
        "I am going to perform logistic regression to predict {target} using the predictors: {predictors}.".format(
            target=target, predictors=predictors
        )
    )
    context.add_code(
        "analyzer.logit(target='{target}', predictors={predictors})".format(
            target=target, predictors=str(parse_predictor_list_from_str(predictors))
        )
    )
    logit_report = context._data_container.analyzer.logit(
        target=target.strip(), predictors=parse_predictor_list_from_str(predictors)
    )
    print_debug(f"_logit_function call finished")
    context.add_table(logit_report.metrics("both"), add_to_vectorstore=False)
    context.add_table(logit_report.coefs(), add_to_vectorstore=False)
    output_str = context.add_dict(logit_report._to_dict())
    output_str += "\n" + context.add_figure(
        logit_report.plot_diagnostics("train"), text_description="Diagnostic plots."
    )
    print_debug(f"_logit_function output: {output_str}")
    return output_str


def build_logit_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_logit_function, context=context),
        name="logit_function",
        description="""Performs logistic regression.
        Returns a JSON string containing coefficients and metrics.
        Also, plots diagnostic plots.""",
        fn_schema=_LogitToolInput,
    )
