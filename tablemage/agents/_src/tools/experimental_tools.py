import pandas as pd
import subprocess
import sys
import tempfile
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.collections import Collection
from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field
from functools import partial
from .tooling_context import ToolingContext
from .tooling_utils import tooling_decorator
from .._debug.logger import print_debug


def to_figure(obj) -> tuple[Figure, bool]:
    """
    Converts a Matplotlib object to a Figure, if possible.
    If the object is not from Matplotlib, it returns the object unchanged.

    Parameters:
        obj: Any Python object.

    Returns:
        A Matplotlib Figure object or the original object.
        Also a boolean indicating whether the output object is a Figure.
    """
    if isinstance(obj, Figure):
        return obj, True

    if isinstance(obj, Axes):
        return obj.figure, True

    if isinstance(obj, Line2D):
        fig, ax = plt.subplots()
        ax.add_line(obj)
        ax.set_xlim(obj.get_xdata().min(), obj.get_xdata().max())
        ax.set_ylim(obj.get_ydata().min(), obj.get_ydata().max())
        return fig, True

    if isinstance(obj, Collection):
        fig, ax = plt.subplots()
        ax.add_collection(obj)
        ax.autoscale_view()
        return fig, True

    return obj, False


class _PythonToolInput(BaseModel):
    code: str = Field(
        description="The Python code to execute. "
        + "The pandas library is already imported. "
        + "The DataFrame is preloaded as `df_all`. "
        + "You MUST save output to `result` variable.",
    )


def python_env_code_run_backend(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    code: str,
):
    """
    Executes a Python code snippet in a separate subprocess with DataFrames preloaded,
    and captures the output data structure.

    Parameters
    ----------
    df_train : pd.DataFrame
        The training DataFrame to preload into the environment.
    df_test : pd.DataFrame
        The test DataFrame to preload into the environment.
    code : str
        The Python code to execute.

    Returns
    -------
    dict
        Contains `result` (deserialized output), `stdout`, `stderr`, and `returncode`.
    """
    sklearn_error_raise_phrases = [
        "MinMaxScaler",
        "StandardScaler",
        "OneHotEncoder",
        "SimpleImputer",
        "KNNImputer",
    ]
    if "sklearn" in code:
        for phrase in sklearn_error_raise_phrases:
            if phrase in code:
                raise ValueError(
                    "ERROR: You should not use scikit-learn transformers in this tool. "
                    "Data transformations are not saved when using this tool. "
                    "Data transformations should be performed with other tools."
                )
    preamble = """\
import pandas as pd
import pickle
import sys
import warnings
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")  # Force a non-GUI backend

# Global variable
result = None

# Define a custom show function
def custom_show():
    global result
    # Get the current figure
    result = plt.gcf()  # Get Current Figure
    plt.close(result)   # Close the figure to avoid showing it accidentally

# Overwrite plt.show
plt.show = custom_show

# Preload the DataFrames
df_train = pd.read_pickle(sys.argv[1])
df_test = pd.read_pickle(sys.argv[2])
df_all = pd.concat([df_train, df_test], axis=0)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
"""

    script_content = (
        preamble
        + "\n"
        + code
        + "\n"
        + """\
# If result is a function, call it to get the result
if callable(result):
    result = result()
        
try:
    with open(sys.argv[3], 'wb') as result_file:
        pickle.dump(result, result_file)
except Exception as e:
    try:
        print(str(result), file=sys.stdout)
    except Exception:
        print('Result could not be serialized or converted to string.', file=sys.stdout)
        print('Try again and print the result to see the output.', file=sys.stdout)
"""
    )

    try:
        # Save the DataFrames to temporary files
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_train:
            df_train.to_pickle(temp_train.name)
            temp_train_path = temp_train.name

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_test:
            df_test.to_pickle(temp_test.name)
            temp_test_path = temp_test.name

        # Create a temporary file for the result
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_result:
            temp_result_path = temp_result.name

        # Save the generated script to a temporary .py file
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_script:
            temp_script.write(script_content.encode())
            temp_script_path = temp_script.name

        # Run the script with the paths to the DataFrames and result file as arguments
        result = subprocess.run(
            [
                sys.executable,
                temp_script_path,
                temp_train_path,
                temp_test_path,
                temp_result_path,
            ],
            capture_output=True,
            text=True,
        )

        # Deserialize the result if the pickle file was created
        output_data = None
        if os.path.exists(temp_result_path):
            try:
                with open(temp_result_path, "rb") as f:
                    output_data = pickle.load(f)
            except Exception as e:
                print_debug(
                    f"An error occurred while deserializing the result: {str(e)}"
                )

        return {
            "result": output_data,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    finally:
        # Clean up temporary files
        for path in [
            temp_train_path,
            temp_test_path,
            temp_result_path,
            temp_script_path,
        ]:
            if os.path.exists(path):
                os.remove(path)


@tooling_decorator
def _python_env_code_run_function(
    code: str,
    context: ToolingContext,
) -> str:
    print_debug(
        "Executing Python code in a separate subprocess with preloaded DataFrames. "
        "Input code: \n" + code
    )
    context.add_thought("I am going to write Python code to solve this problem.")
    context.add_code(
        "df_train, df_test, df_all = analyzer.df_train(), analyzer.df_test(), analyzer.df_all()"
    )
    context.add_code(code)
    df_train = context.data_container.analyzer.df_train()
    df_test = context.data_container.analyzer.df_test()

    if df_train.index.equals(df_test.index):
        context.add_thought(
            "The train and test DataFrames have the same index. "
            "I will replace the test DataFrame with an empty DataFrame before "
            "executing the Python code."
        )
        # make empty DataFrame suitable for concatenation
        df_test = pd.DataFrame(columns=df_test.columns)

    try:
        result = python_env_code_run_backend(
            df_train=df_train,
            df_test=df_test,
            code=code,
        )
    except Exception as e:
        print_debug(
            f"ERROR: An error occurred while executing the Python code. "
            f"The error message is: {str(e)}."
        )
        raise e

    result_actual, is_figure = to_figure(result["result"])

    print_debug(
        f"Python code execution completed. "
        f"Output: {result['stdout']}, "
        f"Error: {result['stderr']}, "
        f"Result: {result_actual}"
    )
    str_to_append = ""
    if is_figure:
        str_to_append = context.add_figure(
            fig=result_actual,
            text_description="The figure output of the Python code execution.",
        )
    elif isinstance(result_actual, pd.DataFrame):
        str_to_append = context.add_table(table=result_actual)
    elif isinstance(result_actual, dict):
        context.add_dict(
            dictionary=result_actual,
            description="The result of the Python code execution. "
            "The Python code was: \n" + code,
        )
    elif isinstance(result_actual, list):
        context.add_str(
            text=str(result_actual),
        )
    elif isinstance(result_actual, str):
        context.add_str(
            text=result_actual,
        )
    elif result_actual is None:
        # try to get the result from the stdout
        result_actual = result["stdout"]
        context.add_thought(f"`{result_actual}`")

    # if everything is empty, return an error message
    if not result["stdout"] and not result["stderr"] and result["result"] is None:
        return "Empty output; please ensure you print or save the result to the `result` variable."

    return (
        f"Output:\n{result['stdout']}\n"
        + f"Error:\n{result['stderr']}\n"
        + f"Result:\n{result['result']}\n"
        + f"Result text:\n{str_to_append}"
    )


python_env_code_run_descr = """\
Use this tool ONLY when no other tools can address the task effectively. \
Useful for: 
- Pandas indexing and operations
- Plotting custom visualizations not supported by other tools
- Statistical analysis not supported by other tools

DESCRIPTION:
- Executes Python code.
- A preloaded DataFrame, `df_all`, serves as the primary dataset for analysis. 
- Optionally, you can work with `df_train` or `df_test` if explicitly required.
- Save the output data structure to the variable `result`.

IMPORTANT:
- ONLY use this tool as a LAST RESORT. Most tasks can be accomplished using other tools.
- Transformations to DataFrames (scaling, imputation, feature engineering) are not saved. \
    DO NOT USE THIS TOOL FOR DATA TRANSFORMATIONS.

EXAMPLES:
1. `result = df_all.head()`
2. `print(df_all['numeric_var'].std())`
3. `fig, ax = plt.subplots()\nax.plot(df_all['numeric_var'])\nresult = fig`
"""


def build_python_env_code_run_tool(context: ToolingContext) -> FunctionTool:
    """Builds a Python code execution tool."""
    return FunctionTool.from_defaults(
        name="python_env_code_function",
        fn=partial(_python_env_code_run_function, context=context),
        description=python_env_code_run_descr,
        fn_schema=_PythonToolInput,
    )
