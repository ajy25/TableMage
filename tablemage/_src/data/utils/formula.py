import pandas as pd
import numpy as np
import re
from typing import Any

from ...display.print_utils import quote_and_color, color_text


def parse_formula(formula: str, df: pd.DataFrame) -> pd.Series:
    """
    Parse a formula string and return a pandas Series with the result.

    Parameters
    ----------
    formula : str
        A formula string to parse. For example, "1x + 2x" would return the sum
        of the columns "1x" and "2x" in the DataFrame as a pandas Series.
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

    df : pd.DataFrame
        A pandas DataFrame containing the data to use in the formula.

    Returns
    -------
    pd.Series
        A pandas Series with the result of the formula.
    """

    def is_safe_name(col: str) -> bool:
        return re.match(r"^[A-Za-z_]\w*$", col) is not None

    def create_safe_column_mapping(df: pd.DataFrame) -> dict[str, str]:
        safe_mapping = {}
        for col in df.columns:
            if is_safe_name(col):
                safe_mapping[col] = col
            else:
                pattern = r"\W+"
                safe_mapping[col] = f"col_{re.sub(pattern, '_', str(col))}"
        return safe_mapping

    def validate_formula(formula: str) -> bool:
        """Validate that the formula only contains allowed operations and patterns."""
        # Define allowed patterns
        allowed_patterns = [
            r"[\w\d]+",  # Column names (including numbers)
            r"\+",  # Addition
            r"-",  # Subtraction
            r"\*\*",  # Exponentiation
            r"\*",  # Multiplication
            r"/",  # Division
            r"\(",  # Opening parenthesis
            r"\)",  # Closing parenthesis
            r"\s+",  # Whitespace
            r"log",  # Logarithm
            r"exp",  # Exponential
            r"\d+\.?\d*",  # Numbers (including decimals)
            r"sqrt",  # Square root
        ]
        allowed_regex = "|".join(allowed_patterns)
        remaining = re.sub(allowed_regex, "", formula)
        return len(remaining.strip()) == 0

    def create_namespace(
        df: pd.DataFrame, col_mapping: dict[str, str]
    ) -> dict[str, Any]:
        """Create a safe namespace for formula evaluation."""
        namespace = {"np": np, "log": np.log, "exp": np.exp, "sqrt": np.sqrt}
        for original_col, safe_col in col_mapping.items():
            namespace[safe_col] = df[original_col]
        return namespace

    if not validate_formula(formula):
        raise ValueError(f"Invalid formula: {formula}")

    col_mapping = create_safe_column_mapping(df)

    safe_formula = formula
    for col in sorted(col_mapping.keys(), key=len, reverse=True):
        safe_formula = safe_formula.replace(col, col_mapping[col])
    namespace = create_namespace(df, col_mapping)

    try:
        result = eval(safe_formula, {"__builtins__": {}}, namespace)
        if not isinstance(result, pd.Series):
            result = pd.Series(result, index=df.index)

        return result
    except Exception as e:
        raise ValueError(
            f"Error evaluating formula '{formula}' "
            f"(transformed to '{safe_formula}'): {str(e)}"
        )


def color_and_quote_formula_vars(formula: str) -> str:
    """Color and quote the variables in a formula string. Uses the
    `quote_and_color` function to quote and color the variables.

    The function identifies variables in mathematical formulas and wraps them
    appropriately while preserving operators and functions.

    Parameters
    ----------
    formula : str
        The formula string to process, e.g. "x1 + log(y2) * 3"

    Returns
    -------
    str
        The formula with variables quoted and colored
    """

    def find_variables(formula: str) -> list[tuple[str, int, int]]:
        """Find all variables in the formula and their positions."""
        var_pattern = r"(?<![a-zA-Z_])[a-zA-Z_]\w*|(?<![a-zA-Z_])\d+[a-zA-Z_]\w*"
        functions = {"log", "exp", "sqrt"}
        matches = []
        for match in re.finditer(var_pattern, formula):
            var = match.group()
            if var not in functions:
                matches.append((var, match.start(), match.end()))
        return sorted(matches, key=lambda x: -x[1])

    variables = find_variables(formula)

    result = formula
    for var, start, end in variables:
        result = result[:start] + quote_and_color(var, "purple") + result[end:]

    parts = result.split("\033[95m")
    colored_parts = []

    for i, part in enumerate(parts):
        if i == 0:
            if part:
                colored_parts.append(color_text(part, "yellow"))
        else:
            purple_and_rest = part.split("\033[0m")
            if len(purple_and_rest) == 2:
                colored_parts.append("\033[95m" + purple_and_rest[0] + "\033[0m")
                if purple_and_rest[1]:
                    colored_parts.append(color_text(purple_and_rest[1], "yellow"))

    return "".join(colored_parts)
