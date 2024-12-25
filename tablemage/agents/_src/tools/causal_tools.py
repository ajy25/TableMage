from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field
from functools import partial
from typing import Literal
from .tooling_utils import tool_try_except_thought_decorator
from .tooling_context import ToolingContext


# effect estimation tool
class _CausalEffectEstimationInput(BaseModel):
    treatment: str = Field(
        description="The treatment variable. Must be binary (0 or 1 valued)."
    )
    outcome: str = Field(description="The outcome variable.")
    confounders: str = Field(description="The confounding variables, comma delimited.")
    effect_type: Literal["ATT", "ATE"] = Field(
        description="The type of effect to estimate. "
        "Either 'ATT' or 'ATE'. "
        "ATT: average treatment effect on the treated. "
        "ATE: average treatment effect."
    )
    method: Literal["ipw_estimator", "ipw_weighted_regression"] = Field(
        description="The method to use for estimation. "
        "Either 'ipw_estimator' or 'ipw_weighted_regression'. "
        "ipw_estimator: inverse probability weighting (IPW) estimator. "
        "ipw_weighted_regression: inverse probability weighting (IPW) weighted regression."
    )


@tool_try_except_thought_decorator
def _estimate_effect(
    treatment: str,
    outcome: str,
    confounders: str,
    effect_type: Literal["ATT", "ATE"],
    method: Literal["ipw_estimator", "ipw_weighted_regression"],
    context: ToolingContext,
) -> str:
    if method == "ipw_estimator":
        method_string = "inverse probability weighting (IPW) estimator"
    elif method == "ipw_weighted_regression":
        method_string = "inverse probability weighting (IPW) weighted regression"
    confounders_as_list = [c.strip() for c in confounders.split(",")]

    if effect_type == "ATT":
        context.add_thought(
            f"I am going to estimate the average treatment effect on the treated "
            f" of {treatment} on {outcome} while controlling for {confounders_as_list} "
            f"with the {method_string} method."
        )
        context.add_code(
            f"analyzer.causal(treatment='{treatment}', outcome='{outcome}', "
            f"confounders='{confounders_as_list}').estimate_att(method='{method}')"
        )
        report = context.data_container.analyzer.causal(
            treatment=treatment, outcome=outcome, confounders=confounders_as_list
        ).estimate_att(method=method)
        output = context.add_dict(report._to_dict())
    elif effect_type == "ATE":
        context.add_thought(
            f"I am going to estimate the average treatment effect of {treatment} "
            f"on {outcome} while controlling for {confounders_as_list} with the {method_string} method."
        )
        context.add_code(
            f"analyzer.causal(treatment='{treatment}', outcome='{outcome}', "
            f"confounders='{confounders_as_list}').estimate_ate(method='{method}')"
        )
        report = context.data_container.analyzer.causal(
            treatment=treatment, outcome=outcome, confounders=confounders_as_list
        ).estimate_ate(method=method)
        output = context.add_dict(report._to_dict())
    return output


def build_estimate_causal_effect_tool(context: ToolingContext):
    return FunctionTool.from_defaults(
        fn=partial(_estimate_effect, context=context),
        name="estimate_causal_effect_tool",
        description="Estimates the causal effect of a treatment on an outcome. "
        "Adjusts for confounders. "
        "Can estimate the average treatment effect on the treated (ATT) "
        "or the average treatment effect (ATE).",
        fn_schema=_CausalEffectEstimationInput,
    )
