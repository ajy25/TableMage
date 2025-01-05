import functools
from .tooling_context import ToolingContext, ToolCall
from .._debug.logger import print_debug
import traceback


def tooling_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        function_name = func.__name__
        if function_name[0] == "_":
            function_name = function_name[1:]

        # get the params, but do not include the context
        params = f"args={args}, kwargs={kwargs}"

        context = kwargs.get("context")
        if not isinstance(context, ToolingContext):
            return func(*args, **kwargs)

        new_toolcall = ToolCall(
            tool_fn_name=function_name,
            tool_fn_args=params,
        )

        print_debug(f"Tool call: {new_toolcall}.")

        if context.is_repeat_toolcall(toolcall=new_toolcall):
            return (
                f"A tool call to `{function_name}` with identical arguments "
                "was repeated. Please use a different tool or change the arguments."
            )
        context.add_toolcall(toolcall=new_toolcall)

        try:
            return func(*args, **kwargs)
        except Exception as e:
            tb_str = "".join(traceback.format_exception(None, e, e.__traceback__))
            print_debug(f"An error occurred: {e}. " f"Traceback: {tb_str}. ")
            thought = f"An error occurred: `{e}"
            if thought[-1] == ".":
                thought = thought[:-1]
            thought += "`. Let me try again."
            context.add_thought(thought)
            return thought

    return wrapper


def parse_str_list_from_str(str_list_str: str) -> list[str]:
    return [elem.strip() for elem in str_list_str.split(",")]


def parse_num_list_from_str(num_list_str: str) -> list[float]:
    return [float(elem.strip()) for elem in num_list_str.split(",")]


def convert_bool_str_to_bool(bool_str: str) -> bool:
    if not isinstance(bool_str, str):
        if isinstance(bool_str, bool):
            return bool_str
        else:
            raise ValueError("bool_str must be a boolean, i.e. True or False.")
    if bool_str.lower() == "true":
        return True
    elif bool_str.lower() == "false":
        return False
    else:
        raise ValueError("bool_str must be a boolean, i.e. True or False.")
