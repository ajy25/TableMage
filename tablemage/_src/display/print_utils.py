from typing import Literal
import sys
import os
import logging
from contextlib import contextmanager
import pandas as pd
from .print_options import print_options


def color_text(
    text, color: Literal["red", "blue", "green", "yellow", "purple", "none"]
) -> str:
    """Returns text in a specified color.

    Parameters
    ----------
    text : str
        The input text.

    color : Literal['red', 'blue', 'green', 'yellow', 'purple', 'none']
        The color of the text.

    Returns
    -------
    str
    """
    # force text to be a string
    text = str(text)

    if color == "none":
        return text
    elif color == "red":
        return "\033[91m" + text + "\033[0m"
    elif color == "blue":
        return "\033[94m" + text + "\033[0m"
    elif color == "purple":
        return "\033[95m" + text + "\033[0m"
    elif color == "green":
        return "\033[92m" + text + "\033[0m"
    elif color == "yellow":
        return "\033[93m" + text + "\033[0m"


def bold_text(text):
    """Returns text in bold."""
    return "\033[1m" + text + "\033[0m"


def strip_ansi(text: str) -> str:
    """Strips ANSI codes from text.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    return (
        text.replace("\033[91m", "")
        .replace("\033[92m", "")
        .replace("\033[93m", "")
        .replace("\033[94m", "")
        .replace("\033[95m", "")
        .replace("\033[1m", "")
        .replace("\033[0m", "")
    )


def print_wrapped(
    text: str,
    type: Literal["WARNING", "UPDATE", "PROGRESS", "NOTE", None] = None,
    level: Literal["INFO", "DEBUG"] = "INFO",
):
    """Logs text.

    Parameters
    ----------
    text : str.
    type : Literal['WARNING', 'UPDATE', 'PROGRESS', None].
        Default: None.
    level : Literal['INFO', 'DEBUG'].
        Default: 'INFO'.
    """
    if print_options._muted:
        return

    if level == "DEBUG":
        print_options._log_debug(strip_ansi(text), secondary=True)
    elif level == "INFO":
        print_options._log_info(strip_ansi(text), secondary=True)

    base_message = text
    if type == "WARNING":
        base_message = color_text("WARN: ", "red") + base_message
    elif type == "UPDATE":
        base_message = color_text("UPDT: ", "green") + base_message
    elif type == "PROGRESS":
        base_message = color_text("PROG: ", "yellow") + base_message
    elif type == "NOTE":
        base_message = color_text("NOTE: ", "yellow") + base_message

    if level == "DEBUG":
        print_options._log_debug(
            fill_ignore_format(base_message, width=print_options._max_line_width)
        )
    elif level == "INFO":
        print_options._log_info(
            fill_ignore_format(base_message, width=print_options._max_line_width)
        )


def list_to_string(
    lst,
    color: Literal["red", "blue", "green", "yellow", "purple", "none"] = "purple",
    include_quotes: bool = True,
) -> str:
    """
    Converts a Python list to a string representation with
    elements separated by commas.

    Parameters
    ----------
    lst : list
        The input list.

    color : Literal['red', 'blue', 'green', 'yellow', 'purple', 'none']
        Default: 'purple'. The color of the elements.

    include_quotes : bool
        Default: True. If True, the elements are enclosed in quotes.

    Returns
    -------
    str
    """
    msg = ""
    for i, elem in enumerate(lst):
        if include_quotes:
            elem = f"'{elem}'"
        if i == len(lst) - 1:
            msg += color_text(elem, color)
        else:
            msg += color_text(elem + ", ", color)
    return msg


def len_ignore_format(text: str) -> int:
    """Returns the length of a string without ANSI codes.

    Parameters
    ----------
    text : str

    Returns
    -------
    int
        The length without ANSI codes.
    """
    base_len = len(text)
    if "\033[91m" in text:
        count = text.count("\033[91m")
        base_len -= 5 * count
    if "\033[92m" in text:
        count = text.count("\033[92m")
        base_len -= 5 * count
    if "\033[93m" in text:
        count = text.count("\033[93m")
        base_len -= 5 * count
    if "\033[94m" in text:
        count = text.count("\033[94m")
        base_len -= 5 * count
    if "\033[95m" in text:
        count = text.count("\033[95m")
        base_len -= 5 * count
    if "\033[1m" in text:
        count = text.count("\033[1m")
        base_len -= 4 * count
    if "\033[0m" in text:
        count = text.count("\033[0m")
        base_len -= 4 * count
    return base_len


def fill_ignore_format_single_line(
    text: str,
    width: int = print_options._max_line_width,
    initial_indent: int = 0,
    subsequent_indent: int = 6,
) -> str:
    """Wraps text to a max width of TOSTR_MAX_WIDTH. Text must NOT
    contain any newline characters.

    Parameters
    ----------
    text : str
        The text to be wrapped.

    width : int
        Default: print_options._max_line_width. The maximum width of the wrapped text.

    initial_indent : int
        Default: 0. The number of spaces to indent the first line.

    subsequent_indent : int
        Default: 6. The number of spaces to indent subsequent lines.

    Returns
    -------
    str
    """
    if "\n" in text:
        raise ValueError("Text must not contain newline characters.")

    text_split = text.split(" ")
    newstr = ""

    newstr += " " * initial_indent
    line_length = initial_indent

    for word in text_split:
        if line_length + len_ignore_format(word) > width:
            newstr += "\n"
            newstr += " " * subsequent_indent
            line_length = subsequent_indent
        newstr += word + " "
        line_length += len_ignore_format(word) + 1

    return newstr


def fill_ignore_format(
    text: str,
    width: int = print_options._max_line_width,
    initial_indent: int = 0,
    subsequent_indent: int = 6,
) -> str:
    """Wraps text to a max width of TOSTR_MAX_WIDTH.

    Parameters
    ----------
    test : str
        The text to be wrapped.

    width : int
        Default: print_options._max_line_width. The maximum width of the wrapped text.

    initial_indent : int
        Default: 0. The number of spaces to indent the first line.

    subsequent_indent : int
        Default: 6. The number of spaces to indent subsequent lines.

    Returns
    -------
    str
    """
    return "\n".join(
        [
            fill_ignore_format_single_line(
                line, width, initial_indent, subsequent_indent
            )
            for line in text.split("\n")
        ]
    )


def quote_and_color(
    text: str,
    color: Literal["red", "blue", "green", "yellow", "purple", "none"] = "blue",
) -> str:
    """Wraps provided text in quotations. Then, colors the result a given color.

    Parameters
    ----------
    text : str

    color : Literal["red", "blue", "green", "yellow", "purple", "none"]
        Default: "blue".

    Returns
    -------
    str
        Transformed text.
    """
    output = f"'{text}'"
    output = color_text(output, color=color)
    return output


@contextmanager
def suppress_std_output():
    """Suppress stdout and stderr."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


@contextmanager
def suppress_print_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        print_options.mute()
        try:
            yield
        finally:
            sys.stdout = old_stdout
            print_options.unmute()


@contextmanager
def suppress_logging(level=logging.FATAL):
    """Temporarily suppress logging output for specified loggers."""
    loggers = ["httpx", "root"]  # List of logger names to suppress
    previous_levels = {}

    # Suppress logging for each logger in the list
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        previous_levels[logger_name] = logger.getEffectiveLevel()
        logger.setLevel(level)

    try:
        yield
    finally:
        # Restore the previous logging levels
        for logger_name, previous_level in previous_levels.items():
            logging.getLogger(logger_name).setLevel(previous_level)


def format_two_column(
    left_text: str, right_text: str, total_len: int = print_options._max_line_width
) -> str:
    """Attempts to reformat two strings in two-column format. If the provided
    strings are two long, simply returns the two strings separated by a newline
    character.

    Parameters
    ----------
    left_text : str

    right_text : str

    total_len : int
        Default: default max line width.

    Returns
    -------
    str
        Reformatted string combining the two input strings in two-column format.
    """

    half_length = int(total_len / 2)
    if (
        len_ignore_format(left_text) >= half_length
        or len_ignore_format(right_text) >= half_length
    ):
        return left_text + "\n" + right_text

    left_buffer = half_length - len_ignore_format(left_text)

    return left_text + " " * left_buffer + right_text


def format_value(
    value: float,
    n_decimals: int = print_options._n_decimals,
    mode: Literal["f", "g", "e"] = "f",
) -> str:
    """Format float values with consistent decimal places."""
    if pd.isna(value):
        return "NA"
    if value == 0:
        return "0." + "0" * n_decimals
    if value == 1:
        return "1." + "0" * n_decimals

    if mode == "g":
        return f"{value:.{n_decimals}g}"
    elif mode == "f":
        return f"{value:.{n_decimals}f}"
    elif mode == "e":
        return f"{value:.{n_decimals}e}"
    else:
        raise ValueError("Invalid mode. Must be 'f' or 'g' or 'e'.")
