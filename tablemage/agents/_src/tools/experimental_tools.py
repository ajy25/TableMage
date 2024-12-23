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
        + "The train and test DataFrames are preloaded as 'df_train' and 'df_test'. "
        + "Feel free to import other standard data science libraries. "
        + "Return the result in the 'result' variable.",
    )


def python_env_code_run_backend(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    code: str,
):
    """
    Executes a Python code snippet in a separate subprocess with a DataFrame named 'df' preloaded,
    and captures the output data structure.

    Parameters
    ----------
    df_train : pd.DataFrame
        The DataFrame to preload into the environment.

    df_test : pd.DataFrame
        The test DataFrame to use for the Analyzer.

    code : str
        The Python code to execute.

    Returns:
        dict: Contains 'result' (deserialized output), 'stdout', 'stderr', and 'returncode'.
    """
    preamble = """
import pandas as pd
import pickle
import sys

# Preload the DataFrames
df_train = pd.read_pickle(sys.argv[1])
df_test = pd.read_pickle(sys.argv[2])

# Placeholder for the result
result = None
"""
    # Append the agent's code and save the result to a pickle file
    script_content = (
        preamble
        + "\n"
        + code
        + "\n"
        + """
# Serialize the result to a file
with open(sys.argv[3], 'wb') as result_file:
    pickle.dump(result, result_file)
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

        # Save the code to a temporary script file
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

        # Deserialize the result
        output_data = None
        if os.path.exists(temp_result_path):
            with open(temp_result_path, "rb") as f:
                output_data = pickle.load(f)

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
    context.add_code("df_train, df_test = analyzer.df_train(), analyzer.df_test()")
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

    result = python_env_code_run_backend(
        df_train=df_train,
        df_test=df_test,
        code=code,
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
            "The Python code did not return a result. " "The Python code was: \n" + code
        )
    return f"StdOut:\n{result['stdout']}\nStdErr:\n{result['stderr']}\nResult:\n{result['result']}"


python_env_code_run_descr = """\
Executes a Python code snippet in a separate subprocess with train DataFrame named 'df_train' preloaded, and \
test DataFrame named 'df_test' preloaded. The result must be saved in the 'result' variable. \
The result must be a number, string, list, dictionary, or DataFrame.
A string containing the stdout, stderr, and str(result) is returned.
Never make plots with this tool.

Example input:
df_all = pd.concat([df_train, df_test], axis=0)
result = df_all.describe()
"""


def build_python_env_code_run_tool(context: ToolingContext) -> FunctionTool:
    """Builds a Python code execution tool."""
    return FunctionTool.from_defaults(
        name="python_env_code_run",
        fn=partial(python_env_code_run, context=context),
        description=python_env_code_run_descr,
        fn_schema=_PythonToolInput,
    )
