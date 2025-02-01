import numpy as np
from ..display.print_options import print_options
from ..display.print_utils import (
    color_text,
    bold_text,
    list_to_string,
    fill_ignore_format,
    format_two_column,
)
from scipy.stats import norm


class CausalReport:
    """Class for storing and displaying causal inference results."""

    def __init__(
        self,
        estimate: float,
        se: float,
        n_units: int,
        n_units_treated: int,
        outcome_var: str,
        treatment_var: str,
        confounders: list[str],
        estimand: str,
        method: str,
        method_description: str,
        p_value: float | None = None,
    ):
        """Initializes a CausalReport object.

        Parameters
        ----------
        estimate : float
            The estimate of the causal effect.

        se : float
            The standard error of the estimator.

        n_units : int
            The number of units in the data.

        n_units_treated : int
            The number of treated units in the data.

        outcome_var : str
            The name of the outcome variable.

        treatment_var : str
            The name of the treatment variable.

        confounders : list[str]
            The names of the confounding variables.

        estimand : str
            The estimand of the causal effect. Either "ate" or "att".

        method : str
            The method used to estimate the causal effect.

        method_description : str
            A description of the method used to estimate the causal effect.
        """

        self._estimate = estimate
        self._estimate_se = se
        self._n_units = n_units
        self._n_units_treated = n_units_treated
        self._outcome_var = outcome_var
        self._treatment_var = treatment_var
        self._confounders = confounders
        self._estimand = estimand
        self._method = method
        self._method_description = method_description

        if p_value is not None:
            self._p_value = p_value
        else:
            # Calculate p-value from standard error
            self._p_value = 2 * (1 - norm.cdf(abs(self._estimate) / self._estimate_se))

    def effect(self):
        """Returns the estimate of the causal effect."""
        return self._estimate

    def se(self):
        """Returns the standard error of the estimator."""
        return self._estimate_se

    def n_units(self):
        """Returns the number of units in the data."""
        return self._n_units

    def pval(self):
        """Returns the p-value of the estimator."""
        return self._p_value

    def _to_dict(self):
        """Converts the CausalReport object to a dictionary."""
        return {
            "estimand": self._estimand,
            "estimate": self._estimate,
            "se": self._estimate_se,
            "p_value": self._p_value,
            "n_units": self._n_units,
            "n_units_treated": self._n_units_treated,
            "outcome_var": self._outcome_var,
            "treatment_var": self._treatment_var,
            "confounders": self._confounders,
            "method": self._method,
            "method_description": self._method_description,
        }

    def __str__(self):
        max_width = print_options._max_line_width
        n_dec = print_options._n_decimals

        top_divider = color_text("=" * max_width, "none") + "\n"
        bottom_divider = "\n" + color_text("=" * max_width, "none")
        divider = "\n" + color_text("-" * max_width, "none") + "\n"
        divider_invisible = "\n" + " " * max_width + "\n"

        title_message = bold_text("Causal Effect Estimation Report")

        estimand = (
            "Avg Trmt Effect (ATE)"
            if self._estimand == "ate"
            else "Avg Trmt Effect on Trtd (ATT)"
        )
        estimate_message = ""
        estimate_message += (
            format_two_column(
                f"{bold_text('Estimate:')} "
                f"{color_text(f'{self._estimate:.{n_dec}f}', 'yellow')}",
                f"{bold_text('Std Err:')} "
                f"{color_text(f'{self._estimate_se:.{n_dec}f}', 'yellow')}",
                max_width,
            )
            + "\n"
        )
        pval_str = f"{self._p_value:.{n_dec}e}"
        if self._p_value <= 0.05:
            pval_color = "green"
        else:
            pval_color = "red"
        estimate_message += format_two_column(
            f"{bold_text('Estimand:')} " f"{color_text(estimand, 'blue')}",
            f"{bold_text('p-value:')} " f"{color_text(pval_str, pval_color)}",
            max_width,
        )

        treatment_message = bold_text("Treatment variable:\n") + color_text(
            "  '" + self._treatment_var + "'", "purple"
        )

        outcome_message = bold_text("Outcome variable:\n") + color_text(
            "  '" + self._outcome_var + "'", "purple"
        )

        confounders_message = bold_text("Confounders:\n")
        if len(self._confounders) == 0:
            confounders_message += fill_ignore_format(
                color_text("None", "yellow"), initial_indent=2, subsequent_indent=2
            )
        else:
            confounders_message += fill_ignore_format(
                list_to_string(self._confounders), initial_indent=2, subsequent_indent=2
            )

        method_message = bold_text("Method:\n") + fill_ignore_format(
            color_text(self._method, "blue"), initial_indent=2, subsequent_indent=2
        )

        return (
            top_divider
            + title_message
            + divider
            + estimate_message
            + divider
            + treatment_message
            + divider_invisible
            + outcome_message
            + divider_invisible
            + confounders_message
            + divider
            + method_message
            + bottom_divider
        )

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text(self.__class__.__name__ + "(...)")
        else:
            p.text(str(self))
