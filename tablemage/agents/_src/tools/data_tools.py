from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field
from functools import partial
from .tooling_utils import tooling_decorator
from .tooling_context import ToolingContext


# pandas query tool
class PandasQueryInput(BaseModel):
    query: str = Field(
        description="The natural language query. "
        + "For example, 'Show me the top 5 rows with higest miles per gallon'."
    )


@tooling_decorator
def _pandas_query_function(query: str, context: ToolingContext) -> str:
    context.add_thought(
        f"I am going to write and run Python code to answer the query: {query}."
    )
    context.add_code("df = analyzer.df_all()")
    response = context._data_container.pd_query_engine.query(query)
    pandas_instructions_str = response.metadata["pandas_instruction_str"]
    context.add_code(pandas_instructions_str)
    raw_pandas_output = response.metadata["raw_pandas_output"]
    context.add_thought(f"The output of the query is:\n{raw_pandas_output}.")
    return str(response)


def build_pandas_query_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_pandas_query_function, context=context),
        name="pandas_query_function",
        description="""Tool for querying the dataset/dataframe using natural language.
The tool will write and run pandas code to answer the query.
Example use cases:
- Filtering rows based on a condition (e.g. largest, smallest).
- Selecting specific columns.
- Finding mean/median/mode of a column.
- Any other operation that can be performed using pandas commands.
The user CANNOT see the output of this query.
New variables will NOT be persisted in the dataset.
USE THIS TOOL AS A LAST RESORT, i.e. only if no other tools are relevant to your desired operation.
        """,
        fn_schema=PandasQueryInput,
    )


# dataset summary tool
class _BlankInput(BaseModel):
    pass


@tooling_decorator
def _dataset_summary_function(context: ToolingContext) -> str:
    context.add_thought(
        "I am going to obtain a summary of the dataset, which includes the shape of the training and test datasets, "
        "as well as the numeric and categorical variables in the dataset."
    )
    context.add_code("analyzer.shape('train')")
    context.add_code("analyzer.shape('test')")
    context.add_code("analyzer.numeric_vars()")
    context.add_code("analyzer.categorical_vars()")
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
    output_dict["numeric_vars"] = (
        context._data_container.analyzer.datahandler().numeric_vars()
    )
    output_dict["categorical_vars"] = (
        context._data_container.analyzer.datahandler().categorical_vars()
    )
    return context.add_dict(output_dict)


def build_dataset_summary_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_dataset_summary_function, context=context),
        name="dataset_summary_function",
        description="Provides a summary of the dataset, "
        "which includes the shape of the training and test datasets, "
        "as well as the names of the numeric and categorical variables in the dataset.",
        fn_schema=_BlankInput(),
    )


# Get variable description tool
class _GetVariableDescriptionInput(BaseModel):
    var: str = Field(description="The variable to get the description of.")


@tooling_decorator
def _get_variable_description_function(var: str, context: ToolingContext) -> str:
    return context._data_container.variable_info.get_description(var)


def build_get_variable_description_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_get_variable_description_function, context=context),
        name="get_variable_description_tool",
        description="Returns the description of a variable. "
        "If no description is available, an empty string will be returned.",
        fn_schema=_GetVariableDescriptionInput,
    )


# Set variable description tool
class _SetVariableDescriptionInput(BaseModel):
    var: str = Field(description="The variable to set the description of.")
    description: str = Field(description="The description of the variable.")


@tooling_decorator
def _set_variable_description_function(
    var: str, description: str, context: ToolingContext
) -> str:
    context._data_container.variable_info.set_description(var, description)
    return f"Description of variable {var} has been set to: {description}."


def build_set_variable_description_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_set_variable_description_function, context=context),
        name="set_variable_description_tool",
        description="Sets the description of a variable. "
        "The 'get_variable_description_tool' tool can be used to retrieve the "
        "description at a later time.",
        fn_schema=_SetVariableDescriptionInput,
    )
