from .tooling_context import ToolingContext


def tool_try_except_thought_decorator(func):
    def wrapper(*args, **kwargs):
        context = kwargs.get("context") or next(
            (arg for arg in args if isinstance(arg, ToolingContext)), None
        )
        try:
            return func(*args, **kwargs)
        except Exception as e:
            thought = f"An error occurred: '{e}'"
            if thought[-1] != ".":
                thought += "."
            thought += " I'll try again."
            context.add_thought(thought)
            raise e

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
