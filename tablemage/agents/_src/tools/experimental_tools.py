import pandas as pd
import subprocess
import sys
import tempfile
import os
import pickle

from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field
from functools import partial
from .tooling_context import ToolingContext
from .tooling_utils import tool_try_except_thought_decorator
from .._debug.logger import print_debug


class _PythonToolInput(BaseModel):
    code: str = Field(
        description="The Python code to execute. "
        + "The pandas library is already imported. "
        + "The DataFrame is preloaded as `df_all`. "
        + "You must print data or save to `result` variable to view.",
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

    # This portion is the Python script preamble. Notice there's no extra indentation
    # inside the triple-quoted string so it can be validly executed as Python code.
    preamble = """\
import pandas as pd
import pickle
import sys
import warnings

# Preload the DataFrames
df_train = pd.read_pickle(sys.argv[1])
df_test = pd.read_pickle(sys.argv[2])
df_all = pd.concat([df_train, df_test], axis=0)

result = None

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
"""

    # Append the user-provided code plus the snippet for serializing the result.
    # Again, ensure that indentation aligns correctly so Python doesn't complain.
    script_content = (
        preamble
        + "\n"
        + code
        + "\n"
        + """# Serialize the result to a file
try:
    with open(sys.argv[3], 'wb') as result_file:
        pickle.dump(result, result_file)
except Exception as e:
    try:
        print(str(e), file=sys.stdout)
    except Exception:
        raise e
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
                print(f"An error occurred while deserializing the result: {str(e)}")

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


@tool_try_except_thought_decorator
def python_env_code_run(
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
            f"An error occurred while executing the Python code. "
            f"The error message is: {str(e)}"
        )

    result_actual = result["result"]
    if isinstance(result_actual, pd.DataFrame):
        context.add_table(table=result_actual)
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
        context.add_thought(
            "The Python code did not return a result. The Python code was: \n" + code
        )

    # if everything is empty, return an error message
    if not result["stdout"] and result["result"] is None:
        return "Empty output; please ensure you print or save the result to the `result` variable."

    return f"StdOut:\n{result['stdout']}\nStdErr:\n{result['stderr']}\nResult:\n{result['result']}"


python_env_code_run_descr = """\
Use this tool ONLY when no other tools can address the task effectively. \
This tool is particularly suited for exploring datasets and \
performing operations using pandas functions.

DESCRIPTION:
- Executes Python code.
- A preloaded DataFrame, `df_all`, serves as the primary dataset for analysis. 
- Optionally, you can work with `df_train` or `df_test` if explicitly required.
- Save the output data structure to the variable `result`. \
    Acceptable types for `result`: dictionary or DataFrame.
- Modifications to the DataFrames are not saved.
- You must explicitly print data structures or save to `result` to view them \
    in the output.

IMPORTANT:
- ONLY use this tool as a LAST RESORT. Most tasks can be accomplished using other tools.
- Do not create plots using this tool.

EXAMPLE INPUTS:
1. `result = df_all.describe()`
2. `result = df_all.head()`
3. `result = df_all['categorical_var'].value_counts()`
4. `print(df_all['numeric_var'].std())`
"""


def build_python_env_code_run_tool(context: ToolingContext) -> FunctionTool:
    """Builds a Python code execution tool."""
    return FunctionTool.from_defaults(
        name="python_env_code_run",
        fn=partial(python_env_code_run, context=context),
        description=python_env_code_run_descr,
        fn_schema=_PythonToolInput,
    )
