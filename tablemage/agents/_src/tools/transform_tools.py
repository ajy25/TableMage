from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field
from functools import partial
from .tooling_context import ToolingContext
from .tooling_utils import (
    tool_try_except_thought_decorator,
    parse_str_list_from_str,
    parse_num_list_from_str,
    convert_bool_str_to_bool,
)


class _ImputeInput(BaseModel):
    vars: str = Field(
        description="A comma delimited string of variables to impute missing values. "
        "An example input (without the quotes) is: 'var1, var2, var3'."
    )
    numeric_strategy: str = Field(
        description="The imputation strategy for numeric variables. "
        "Options are: 'mean', 'median', '5nn', and '10nn'."
    )
    categorical_strategy: str = Field(
        description="The imputation strategy for categorical variables. "
        "Options are: 'most_frequent' and 'missing'. "
        "Note that 'missing' will create a new category for missing values."
    )


@tool_try_except_thought_decorator
def impute_function(
    vars: str, numeric_strategy: str, categorical_strategy: str, context: ToolingContext
) -> str:
    context.add_thought(
        "I am going to impute missing values in the dataset using the following strategies: "
        f"numeric: {numeric_strategy}, categorical: {categorical_strategy}."
    )
    context.add_code(
        f"analyzer.impute(include_vars={vars}, numeric_strategy={numeric_strategy}, "
        f"categorical_strategy={categorical_strategy})"
    )
    vars_list = parse_str_list_from_str(vars)
    context.data_container.analyzer.impute(
        include_vars=vars_list,
        numeric_strategy=numeric_strategy,
        categorical_strategy=categorical_strategy,
    )
    context.data_container.update_df()
    return "Imputation complete."


def build_impute_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(impute_function, context=context),
        name="impute_tool",
        description="""This tool allows you to impute missing values in the dataset using the specified strategies.""",
        fn_schema=_ImputeInput,
    )


class _DropHighlyMissingVarsInput(BaseModel):
    threshold: float = Field(
        description="Proportion of missing values above which a column is dropped. "
        "For example, if threshold = 0.2, then columns with more than 20% missing "
        "values are dropped."
    )
    ignore_vars: str = Field(
        description="A comma delimited string of variables to ignore when dropping columns. "
        "An example input (without the quotes) is: 'var1, var2, var3'."
    )


@tool_try_except_thought_decorator
def drop_highly_missing_vars_function(
    threshold: float, ignore_vars: str, context: ToolingContext
) -> str:
    threshold = float(threshold)

    context.add_thought(
        "I am going to drop columns with a proportion of missing values above the threshold "
        f"{threshold}."
    )
    context.add_code(
        f"analyzer.drop_highly_missing_vars(threshold={threshold}, ignore_vars={ignore_vars})"
    )
    ignore_vars_list = parse_str_list_from_str(ignore_vars)

    cols_before_drop = context.data_container.analyzer.vars()

    context.data_container.analyzer.drop_highly_missing_vars(
        threshold=threshold, exclude_vars=ignore_vars_list
    )

    cols_after_drop = context.data_container.analyzer.vars()

    dropped_cols = set(cols_before_drop) - set(cols_after_drop)

    context.data_container.update_df()

    return (
        "Columns with high missing values have been dropped: "
        + ", ".join(dropped_cols)
        + "."
    )


def build_drop_highly_missing_vars_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(drop_highly_missing_vars_function, context=context),
        name="drop_highly_missing_vars_tool",
        description="""This tool allows you to drop columns with a proportion of missing values above a specified threshold.""",
        fn_schema=_DropHighlyMissingVarsInput,
    )


class _SaveStateInput(BaseModel):
    state_name: str = Field(description="The name of the state to save.")


@tool_try_except_thought_decorator
def save_state_function(state_name: str, context: ToolingContext) -> str:
    context.add_thought(
        f"I am going to save the current state of the dataset as {state_name}."
    )
    context.add_code(f"analyzer.save_data_checkpoint({state_name})")
    context.data_container.analyzer.save_data_checkpoint(state_name)
    context.data_container.update_df()
    return f"State {state_name} saved."


def build_save_state_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(save_state_function, context=context),
        name="save_state_tool",
        description="""This tool allows you to save the current state of the dataset.""",
        fn_schema=_SaveStateInput,
    )


class _LoadStateInput(BaseModel):
    state_name: str = Field(description="The name of the dataset state to load.")


@tool_try_except_thought_decorator
def load_state_function(state_name: str, context: ToolingContext) -> str:
    context.add_thought(
        f"I am going to load the state of the dataset saved as {state_name}."
    )
    context.add_code(f"analyzer.load_data_checkpoint({state_name})")
    context.data_container.analyzer.load_data_checkpoint(state_name)
    context.data_container.update_df()
    return f"State {state_name} loaded."


def build_load_state_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(load_state_function, context=context),
        name="load_state_tool",
        description="""This tool allows you to load a previously saved state of the dataset.""",
        fn_schema=_LoadStateInput,
    )


class _BlankInput(BaseModel):
    pass


@tool_try_except_thought_decorator
def revert_to_original_function(context: ToolingContext) -> str:
    context.add_thought("I am going to revert the dataset to its original state.")
    context.add_code("analyzer.load_data_checkpoint()")
    context.data_container.analyzer.load_data_checkpoint()
    context.data_container.update_df()
    return "Dataset reverted to original state."


def build_revert_to_original_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(revert_to_original_function, context=context),
        name="revert_to_original_tool",
        description="""This tool allows you to revert the dataset to its original state.""",
        fn_schema=_BlankInput,
    )


class _EngineerNumericFeatureInput(BaseModel):
    feature_name: str = Field(description="The name of the new feature to engineer.")
    formula: str = Field(
        description="""
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
formula, then the i-th unit of the new feature will be missing."""
    )


@tool_try_except_thought_decorator
def _engineer_numeric_feature_function(
    feature_name: str, formula: str, context: ToolingContext
) -> str:
    context.add_thought(
        f"I am going to engineer a new variable named {feature_name} using the formula: {formula}."
    )
    context.add_code(
        f"analyzer.engineer_numeric_var(name='{feature_name}', formula='{formula}')"
    )
    context.data_container.analyzer.engineer_numeric_var(
        name=feature_name, formula=formula
    )
    context.data_container.update_df()
    return f"Feature {feature_name} engineered."


def build_engineer_numeric_feature_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_engineer_numeric_feature_function, context=context),
        name="engineer_numeric_feature_tool",
        description="This tool allows you to engineer a new numeric variable "
        "using a formula of other numeric variables.",
        fn_schema=_EngineerNumericFeatureInput,
    )


class _EngineerCategoricalFeatureInput(BaseModel):
    feature_name: str = Field(description="The name of the new feature to engineer.")
    numeric_var: str = Field(
        description="The numeric variable to use for engineering the new feature."
    )
    level_names: str = Field(
        description="A comma delimited string of level names for the new categorical feature. "
        "The first level is the lowest level, and the last level is the highest level. "
        "There must be one more level name than the number of thresholds."
    )
    thresholds: str = Field(
        description="A comma delimited string of upper cutoffs/thresholds for the new categorical feature. "
        "The thresholds must be in ascending order. "
        "For example, if thresholds = '0, 10, 20', "
        "and level_names = 'Low, Medium, High, Very High', "
        "then the new variable will have the following levels: "
        "Low (x < 0), Medium (0 <= x < 10), High (10 <= x < 20), Very High (x >= 20)."
    )
    leq: bool = Field(
        description="Boolean for whether the upper cutoffs/thresholds are inclusive "
        "(True) or exclusive (False)."
    )


@tool_try_except_thought_decorator
def _engineer_categorical_feature_function(
    feature_name: str,
    numeric_var: str,
    level_names: str,
    thresholds: str,
    leq: bool,
    context: ToolingContext,
) -> str:
    leq = convert_bool_str_to_bool(leq)

    level_names_list = parse_str_list_from_str(level_names)
    thresholds_list = parse_num_list_from_str(thresholds)

    assert (
        len(level_names_list) == len(thresholds_list) + 1,
        "There must be one more level name than the number of thresholds.",
    )

    thresholds_str = ""
    for i, level in enumerate(level_names_list):
        if i == len(level_names_list) - 1:
            threshold = thresholds_list[i - 1]
            if leq:
                thresholds_str += f"{level} > {threshold}."
            else:
                thresholds_str += f"{level} >= {threshold}."
        else:
            threshold = thresholds_list[i]
            if leq:
                thresholds_str += f"{level} <= {threshold}, "
            else:
                thresholds_str += f"{level} < {threshold}, "

    context.add_thought(
        f"I am going to engineer a new categorical feature named {feature_name} using the numeric variable {numeric_var} as follows: "
        f"{thresholds_str}"
    )
    context.add_code(
        f"analyzer.engineer_categorical_var(name='{feature_name}', numeric_var='{numeric_var}', "
        f"level_names={level_names_list}, thresholds={thresholds_list}, leq={leq})"
    )
    context.data_container.analyzer.engineer_categorical_var(
        name=feature_name,
        numeric_var=numeric_var,
        level_names=level_names_list,
        thresholds=thresholds_list,
        leq=leq,
    )
    context.data_container.update_df()
    return f"Feature {feature_name} engineered."


def build_engineer_categorical_feature_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_engineer_categorical_feature_function, context=context),
        name="engineer_categorical_feature_tool",
        description="""This tool allows you to engineer a new categorical variable "
        "from a numeric variable.""",
        fn_schema=_EngineerCategoricalFeatureInput,
    )


class _OnehotEncodeInput(BaseModel):
    vars: str = Field(
        description="A comma delimited string of variables to one-hot encode. "
        "An example input (without the quotes) is: 'var1, var2, var3'."
    )
    dropfirst: bool = Field(
        description="Whether to drop the first level of the one-hot encoded variables."
    )


@tool_try_except_thought_decorator
def onehot_encode_function(vars: str, dropfirst: bool, context: ToolingContext) -> str:
    dropfirst = convert_bool_str_to_bool(dropfirst)

    context.add_thought(
        "I am going to one-hot encode the following variables: " + vars + "."
    )
    context.add_code(f"analyzer.onehot_encode(include_vars={vars})")

    vars_list = parse_str_list_from_str(vars)
    context.data_container.analyzer.onehot(include_vars=vars_list, dropfirst=dropfirst)
    context.data_container.update_df()
    return "One-hot encoding complete."


def build_onehot_encode_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(onehot_encode_function, context=context),
        name="onehot_encode_tool",
        description="This tool allows you to one-hot encode variables in the dataset.",
        fn_schema=_OnehotEncodeInput,
    )


class _DropNaInput(BaseModel):
    vars: str = Field(
        description="A comma delimited string of variables to drop rows with missing values. "
        "An example input (without the quotes) is: 'var1, var2, var3'."
    )


@tool_try_except_thought_decorator
def drop_na_function(vars: str, context: ToolingContext) -> str:
    context.add_thought(
        "I am going to drop rows with missing values in the following variables: "
        + vars
        + "."
    )
    context.add_code(f"analyzer.dropna(include_vars={vars})")

    vars_list = parse_str_list_from_str(vars)
    context.data_container.analyzer.dropna(include_vars=vars_list)
    context.data_container.update_df()
    return "Rows with missing values dropped."


def build_drop_na_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(drop_na_function, context=context),
        name="drop_na_tool",
        description="""This tool allows you to drop rows with missing values in the dataset.""",
        fn_schema=_DropNaInput,
    )


class _ScaleInput(BaseModel):
    vars: str = Field(
        description="A comma delimited string of variables to scale. "
        "An example input (without the quotes) is: 'var1, var2, var3'."
    )
    method: str = Field(
        description="The scaling method to use. Options are: 'standardize' and 'minmax'."
    )


@tool_try_except_thought_decorator
def scale_function(vars: str, method: str, context: ToolingContext) -> str:
    context.add_thought("I am going to scale the following variables: " + vars + ".")
    context.add_code(f"analyzer.scale(include_vars={vars}, strategy={method})")
    vars_list = parse_str_list_from_str(vars)
    context.data_container.analyzer.scale(include_vars=vars_list, strategy=method)
    context.data_container.update_df()
    return "Scaling complete."


def build_scale_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(scale_function, context=context),
        name="scale_tool",
        description="""This tool allows you to scale numeric variables in the dataset.""",
        fn_schema=_ScaleInput,
    )
