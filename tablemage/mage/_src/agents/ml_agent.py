from llama_index.core.agent import FunctionCallingAgent, ReActAgent
from llama_index.core.llms.function_calling import FunctionCallingLLM

from .utils import build_function_calling_agent
from ..tools.ml_tools import build_ml_regression_tool, build_ml_classification_tool
from ..tools.tooling_context import ToolingContext

from .prompt.ml_agent_system_prompt import ML_SYSTEM_PROMPT


def build_ml_agent(
    llm: FunctionCallingLLM,
    context: ToolingContext,
    system_prompt: str = ML_SYSTEM_PROMPT,
    react: bool = False,
) -> FunctionCallingAgent | ReActAgent:
    """Builds a machine learning agent.

    Parameters
    ----------
    llm : FunctionCallingLLM
        Function calling LLM

    context : ToolingContext
        Tooling context

    system_prompt : str
        System prompt. Default linear regression system prompt is used if not provided.

    react : bool
        If True, a ReActAgent is returned. Otherwise, a FunctionCallingAgent is returned.
        If True, the system prompt is not considered.

    Returns
    -------
    FunctionCallingAgent | ReActAgent
        Either a FunctionCallingAgent or a ReActAgent
    """
    tools = [
        build_ml_regression_tool(context),
        build_ml_classification_tool(context),
    ]
    return build_function_calling_agent(
        llm=llm,
        tools=tools,
        system_prompt=system_prompt,
        react=react,
    )
