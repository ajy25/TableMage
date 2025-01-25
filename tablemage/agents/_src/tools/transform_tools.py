from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field
from functools import partial
from .tooling_context import ToolingContext
from .tooling_utils import (
    tooling_decorator,
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


@tooling_decorator
def _impute_function(
    vars: str,
    numeric_strategy: str = "5nn",
    categorical_strategy: str = "missing",
    context: ToolingContext = None,
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
    return "The dataset has been transformed: missing values have been imputed."


def build_impute_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_impute_function, context=context),
        name="impute_function",
        description="Imputes missing values in the dataset with the specified strategies.",
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


@tooling_decorator
def _drop_highly_missing_vars_function(
    threshold: float = 0.2,
    ignore_vars: str = "",
    context: ToolingContext = None,
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
        "The dataset has been transformed: "
        + "columns with high missing values have been dropped: "
        + ", ".join(dropped_cols)
        + ". "
        + "All variables in current dataset state: "
        + ", ".join(cols_after_drop)
        + "."
    )


def build_drop_highly_missing_vars_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_drop_highly_missing_vars_function, context=context),
        name="drop_highly_missing_vars_function",
        description="Drops columns with a proportion of missing values above a specified threshold.",
        fn_schema=_DropHighlyMissingVarsInput,
    )


class _SaveStateInput(BaseModel):
    state_name: str = Field(description="The name of the state to save.")


@tooling_decorator
def _save_state_function(state_name: str, context: ToolingContext) -> str:
    context.add_thought(
        f"I am going to save the current state of the dataset as {state_name}."
    )
    context.add_code(f"analyzer.save_data_checkpoint({state_name})")
    context.data_container.analyzer.save_data_checkpoint(state_name)
    context.data_container.update_df()
    return f"Dataset state {state_name} saved."


def build_save_state_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_save_state_function, context=context),
        name="save_state_function",
        description="This tool allows you to save the current state of the dataset. "
        "The state can be loaded later.",
        fn_schema=_SaveStateInput,
    )


class _LoadStateInput(BaseModel):
    state_name: str = Field(description="The name of the dataset state to load.")


@tooling_decorator
def _load_state_function(state_name: str, context: ToolingContext) -> str:
    context.add_thought(
        f"I am going to load the state of the dataset saved as {state_name}."
    )
    context.add_code(f"analyzer.load_data_checkpoint({state_name})")
    context.data_container.analyzer.load_data_checkpoint(state_name)
    context.data_container.update_df()
    output_dict = {}
    output_dict["train_shape"] = {
        "n_rows": context._data_container.analyzer.datahandler().df_train().shape[0],
        "n_vars": context._data_container.analyzer.datahandler().df_train().shape[1],
        "n_categorical_vars": len(
            context._data_container.analyzer.datahandler().categorical_vars()
        ),
        "n_numeric_vars": len(
            context._data_container.analyzer.datahandler().numeric_vars()
        ),
    }
    output_dict["test_shape"] = {
        "n_rows": context._data_container.analyzer.datahandler().df_test().shape[0],
        "n_vars": context._data_container.analyzer.datahandler().df_test().shape[1],
        "n_categorical_vars": len(
            context._data_container.analyzer.datahandler().categorical_vars()
        ),
        "n_numeric_vars": len(
            context._data_container.analyzer.datahandler().numeric_vars()
        ),
    }
    output_dict[
        "numeric_vars"
    ] = context._data_container.analyzer.datahandler().numeric_vars()
    output_dict[
        "categorical_vars"
    ] = context._data_container.analyzer.datahandler().categorical_vars()
    return f"Dataset state {state_name} loaded. Dataset summary: " + str(output_dict)


def build_load_state_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_load_state_function, context=context),
        name="load_state_function",
        description="This tool allows you to load a previously saved dataset state.",
        fn_schema=_LoadStateInput,
    )


class _BlankInput(BaseModel):
    pass


@tooling_decorator
def _revert_to_original_function(context: ToolingContext) -> str:
    context.add_thought("I am going to revert the dataset to its original state.")
    context.add_code("analyzer.load_data_checkpoint()")
    context.data_container.analyzer.load_data_checkpoint()
    context.data_container.update_df()
    output_dict = {}
    output_dict["train_shape"] = {
        "n_rows": context._data_container.analyzer.datahandler().df_train().shape[0],
        "n_vars": context._data_container.analyzer.datahandler().df_train().shape[1],
        "n_categorical_vars": len(
            context._data_container.analyzer.datahandler().categorical_vars()
        ),
        "n_numeric_vars": len(
            context._data_container.analyzer.datahandler().numeric_vars()
        ),
    }
    output_dict["test_shape"] = {
        "n_rows": context._data_container.analyzer.datahandler().df_test().shape[0],
        "n_vars": context._data_container.analyzer.datahandler().df_test().shape[1],
        "n_categorical_vars": len(
            context._data_container.analyzer.datahandler().categorical_vars()
        ),
        "n_numeric_vars": len(
            context._data_container.analyzer.datahandler().numeric_vars()
        ),
    }
    output_dict[
        "numeric_vars"
    ] = context._data_container.analyzer.datahandler().numeric_vars()
    output_dict[
        "categorical_vars"
    ] = context._data_container.analyzer.datahandler().categorical_vars()
    return "Dataset reverted to original state. Dataset summary: " + str(output_dict)


def build_revert_to_original_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_revert_to_original_function, context=context),
        name="revert_to_original_function",
        description="This tool allows you to revert the dataset to its original state.",
        fn_schema=_BlankInput,
    )


class _EngineerNumericFeatureInput(BaseModel):
    feature_name: str = Field(description="The name of the new feature to engineer.")
    formula: str = Field(
        description="""\
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


@tooling_decorator
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
    return "The dataset has been transformed: " + f"Feature {feature_name} engineered."


def build_engineer_numeric_feature_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_engineer_numeric_feature_function, context=context),
        name="engineer_numeric_feature_function",
        description="""\
Defines/engineers/makes a new numeric variable as a formula of other numeric variables.
Example Call:
{feature_name: 'new_feature', formula: 'x1 + x2'} 
Example Functionality: Creates a new feature that is the sum of x1 and x2.
""",
        fn_schema=_EngineerNumericFeatureInput,
    )


class _EngineerCategoricalFeatureInput(BaseModel):
    feature_name: str = Field(description="The name of the new feature to engineer.")
    numeric_var: str = Field(
        description="The numeric variable to use for engineering the new categorical variable."
    )
    level_names: str = Field(
        description="A comma-delimited string of names for the levels of the new categorical variable. "
        "There must be one more level name than the number of thresholds."
    )
    thresholds: str = Field(
        description="A comma-delimited string of numeric thresholds for creating the categorical variable levels. "
        "Thresholds must be specified in ascending order."
    )
    leq: bool = Field(
        description="""\
Specifies how the boundaries of the levels are defined.
If True, levels are inclusive on the upper end of a threshold.

For example, with thresholds = '0, 10', level_names = 'Low, Medium, High', \
and leq = True, the levels are:
Low (x <= 0), Medium (0 < x <= 10), High (x > 10).

If leq = False, the levels are:
Low (x < 0), Medium (0 <= x < 10), High (x >= 10).
"""
    )


@tooling_decorator
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
        f"I am going to engineer a new categorical variable named {feature_name} "
        f"using the numeric variable {numeric_var} as follows: "
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
    return (
        "The dataset has been transformed: "
        + f"Feature {feature_name} engineered with thresholds {thresholds_str}."
    )


def build_engineer_categorical_feature_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_engineer_categorical_feature_function, context=context),
        name="engineer_categorical_feature_function",
        description="""\
Engineers a new categorical variable from a numeric variable. \
The new variable is created based on specified thresholds.

-- Example --
Task: 
Create a new categorical variable 'new_feature' from numeric variable 'x1' \
with label 'High' if 'x1' is at least 10, 'Low' otherwise.

Function Call:
{feature_name: 'new_feature', numeric_var: 'x1', level_names: 'Low, High', thresholds: '10', leq: False}

Functionality: 
Creates a new categorical variable with levels 'Low' (x1 < 10) and 'High' (x1 >= 10).
""",
        fn_schema=_EngineerCategoricalFeatureInput,
    )


class _ForceBinaryInput(BaseModel):
    var: str = Field(description="The variable to force to binary (0 or 1).")
    pos_label: str = Field(
        description="The positive label to use for the binary variable. "
        "If empty string, the positive label will be the most frequent value."
    )


@tooling_decorator
def _force_binary_function(var: str, pos_label: str, context: ToolingContext) -> str:
    context.add_thought("I am going to force the variable " + var + " to binary.")
    context.add_code(
        f"analyzer.force_binary(var='{var}', pos_label='{pos_label}', ignore_multiclass=True, rename=True)"
    )
    if pos_label == "":
        pos_label = (
            context.data_container.analyzer.datahandler()
            .df_train()[var]
            .value_counts()
            .idxmax()
        )
    context.data_container.analyzer.force_binary(
        var=var, pos_label=pos_label, ignore_multiclass=True, rename=True
    )
    context.data_container.update_df()
    new_var_name = var + "::" + pos_label

    return (
        "The dataset has been transformed: "
        + f"Variable {var} forced to binary with positive label {pos_label}. "
        + f"The new binary variable is named {new_var_name}. "
        + f"The variable {var} has been removed from the dataset."
    )


def build_force_binary_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_force_binary_function, context=context),
        name="force_binary_function",
        description="Forces a variable to binary numeric (0 or 1).",
        fn_schema=_ForceBinaryInput,
    )


class _OnehotEncodeInput(BaseModel):
    vars: str = Field(
        description="A comma delimited string of variables to one-hot encode. "
        "An example input (without the quotes) is: 'var1, var2, var3'."
    )
    dropfirst: bool = Field(
        description="Whether to drop the first level of the one-hot encoded variables."
    )


@tooling_decorator
def _onehot_encode_function(vars: str, dropfirst: bool, context: ToolingContext) -> str:
    dropfirst = convert_bool_str_to_bool(dropfirst)

    context.add_thought(
        "I am going to one-hot encode the following variables: " + vars + "."
    )
    context.add_code(f"analyzer.onehot(include_vars={vars}, dropfirst={dropfirst})")

    vars_list = parse_str_list_from_str(vars)
    context.data_container.analyzer.onehot(include_vars=vars_list, dropfirst=dropfirst)
    context.data_container.update_df()
    return (
        "The dataset has been transformed: "
        + f"Variables {vars_list} one-hot encoded. "
        + f"The original variables are retained. "
        + f"Drop first level/category (True or False)? {dropfirst}. "
        + f"All variables in current dataset state: "
        + f"{context.data_container.analyzer.vars()}."
    )


def build_onehot_encode_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_onehot_encode_function, context=context),
        name="onehot_encode_function",
        description="One-hot encodes variables in the dataset. "
        "Does not overwrite original variables.",
        fn_schema=_OnehotEncodeInput,
    )


class _DropNaInput(BaseModel):
    vars: str = Field(
        description="A comma delimited string of variables to drop rows with missing values. "
        "An example input (without the quotes) is: 'var1, var2, var3'."
    )


@tooling_decorator
def _drop_na_function(vars: str, context: ToolingContext) -> str:
    context.add_thought(
        "I am going to drop rows with missing values in the following variables: "
        + vars
        + "."
    )
    context.add_code(f"analyzer.dropna(include_vars={vars})")

    vars_list = parse_str_list_from_str(vars)
    train_shape_before = context.data_container.analyzer.datahandler().df_train().shape
    test_shape_before = context.data_container.analyzer.datahandler().df_test().shape
    context.data_container.analyzer.dropna(include_vars=vars_list)
    context.data_container.update_df()
    train_shape_after = context.data_container.analyzer.datahandler().df_train().shape
    test_shape_after = context.data_container.analyzer.datahandler().df_test().shape
    rows_dropped_train = train_shape_before[0] - train_shape_after[0]
    rows_dropped_test = test_shape_before[0] - test_shape_after[0]
    return (
        "The dataset has been transformed: "
        + f"{rows_dropped_train + rows_dropped_test} rows with missing values dropped. "
        + f"Train shape before: {train_shape_before}, after: {train_shape_after}. "
        + f"Test shape before: {test_shape_before}, after: {test_shape_after}."
    )


def build_drop_na_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_drop_na_function, context=context),
        name="drop_na_function",
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


@tooling_decorator
def _scale_function(vars: str, method: str, context: ToolingContext) -> str:
    context.add_thought("I am going to scale the following variables: " + vars + ".")
    context.add_code(f"analyzer.scale(include_vars={vars}, strategy={method})")
    vars_list = parse_str_list_from_str(vars)
    context.data_container.analyzer.scale(include_vars=vars_list, strategy=method)
    context.data_container.update_df()
    return (
        "The dataset has been transformed: "
        + f"Variables {vars_list} scaled using the {method} method."
    )


def build_scale_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_scale_function, context=context),
        name="scale_function",
        description="Scales numeric variable values in the dataset based on a specified strategy. "
        + "Standardization (scale to mean = 0, std = 1) and min-max scaling are supported.",
        fn_schema=_ScaleInput,
    )
