from llama_index.core.agent import (
    FunctionCallingAgent,
    ReActAgent,
)
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core import VectorStoreIndex
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.objects import ObjectIndex
from llama_index.core.schema import QueryBundle
from llama_index.core.memory import (
    ChatMemoryBuffer,
    VectorMemory,
    SimpleComposableMemory,
)

from llama_index.core import Settings

from typing import Literal
from pathlib import Path

from ...._src.display.print_utils import suppress_logging
from .._debug.logger import print_debug

from ..tools.ml_tools import (
    build_ml_regression_tool,
    build_ml_classification_tool,
    build_feature_selection_tool,
    build_clustering_tool,
)
from ..tools.eda_tools import (
    build_test_ttest_tool,
    build_test_anova_tool,
    build_test_chi2_tool,
    build_test_normality_tool,
    build_plot_tool,
    build_plot_pairs_tool,
    build_numeric_summary_statistics_tool,
    build_categorical_summary_statistics_tool,
    build_correlation_comparison_tool,
    build_correlation_matrix_tool,
)
from ..tools.linear_regression_tools import build_ols_tool, build_logit_tool
from ..tools.data_tools import build_dataset_summary_tool
from ..tools.transform_tools import (
    build_drop_highly_missing_vars_tool,
    build_drop_na_tool,
    build_engineer_numeric_feature_tool,
    build_engineer_categorical_feature_tool,
    build_impute_tool,
    build_scale_tool,
    build_onehot_encode_tool,
    build_revert_to_original_tool,
)
from ..tools.experimental_tools import build_python_env_code_run_tool
from ..tools.causal_tools import build_estimate_causal_effect_tool
from ..tools.tooling_context import ToolingContext

from .prompt.single_agent_system_prompt import DEFAULT_SYSTEM_PROMPT


io_path = Path(__file__).resolve().parent.parent / "io"


def build_agent(
    llm: FunctionCallingLLM,
    context: ToolingContext,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    memory: Literal["buffer", "vector"] = "vector",
    memory_size: int = 3000,
    tool_rag: bool = True,
    tool_rag_top_k: int = 5,
    react: bool = False,
    python_only: bool = False,
    tools_only: bool = False,
) -> FunctionCallingAgent | ReActAgent:
    """Builds an agent.

    Parameters
    ----------
    llm : FunctionCallingLLM
        Function calling LLM

    context : ToolingContext
        Tooling context

    system_prompt : str
        System prompt. Default linear regression system prompt is used if not provided.

    memory : Literal["buffer", "vector"]
        Memory type to use. Default is "vector".

    memory_size : int
        Memory size (buffer token limit). Default is 3000.

    tool_rag : bool
        If True, uses RAG for tool retrieval. Default is True.
        Otherwise, includes all tools.

    tool_rag_top_k : int
        Top k tools to retrieve using RAG. Default is 5.
        Ignored if tool_rag is False.
        Tool retreival should be used for the following reasons:
        - Conserve context window space.
        - Improve response time.
        But, tool retrieval results in decreased accuracy,
        as the relevant tools may not be retrieved. Retrieval is done
        based on the user query, not the agent's understanding of the context.

    react : bool
        If True, a ReActAgent is returned. Otherwise, a FunctionCallingAgent is returned.
        If True, the system prompt is not considered.

    python_only : bool
        If True, only the Python environment is provided. Default is False.

    tools_only : bool
        If True, only the tools are provided.
        Otherwise, the Python environment is also provided, ignoring RAG.
        Default is False.

    Returns
    -------
    FunctionCallingAgent | ReActAgent
        Either a FunctionCallingAgent or a ReActAgent
    """

    if memory == "buffer":
        memory_obj = ChatMemoryBuffer.from_defaults(token_limit=memory_size)

    elif memory == "vector":
        vector_store, _ = context.storage_manager.setup_vector_store(
            path=io_path / "_vector_memory",
        )
        buffer_memory = ChatMemoryBuffer.from_defaults(token_limit=memory_size)
        vector_memory = VectorMemory.from_defaults(
            vector_store=vector_store,
            embed_model=Settings.embed_model,
            retriever_kwargs={"similarity_top_k": 1},
        )
        memory_obj = SimpleComposableMemory(
            primary_memory=buffer_memory,
            secondary_memory_sources=[vector_memory],
        )

    else:
        raise ValueError("The memory type must be either 'buffer' or 'vector'.")

    dataset_summary_tool = build_dataset_summary_tool(context)
    summary_str = str(dataset_summary_tool.call())
    print_debug(f"Dataset summary: {summary_str}")
    memory_obj.put(
        ChatMessage.from_str(
            content="Dataset summary: " + summary_str,
            role=MessageRole.SYSTEM,
        )
    )

    if python_only:
        tools = []

    else:
        tools = [
            build_estimate_causal_effect_tool(context),
            build_feature_selection_tool(context),
            build_ml_regression_tool(context),
            build_ml_classification_tool(context),
            build_clustering_tool(context),
            build_test_ttest_tool(context),
            build_test_anova_tool(context),
            build_test_chi2_tool(context),
            build_test_normality_tool(context),
            build_plot_tool(context),
            build_plot_pairs_tool(context),
            build_numeric_summary_statistics_tool(context),
            build_categorical_summary_statistics_tool(context),
            build_correlation_comparison_tool(context),
            build_correlation_matrix_tool(context),
            build_ols_tool(context),
            build_logit_tool(context),
            build_drop_highly_missing_vars_tool(context),
            build_drop_na_tool(context),
            build_engineer_numeric_feature_tool(context),
            build_engineer_categorical_feature_tool(context),
            build_impute_tool(context),
            build_scale_tool(context),
            build_onehot_encode_tool(context),
            dataset_summary_tool,
        ]

    if python_only:
        tools_to_persist = [
            build_python_env_code_run_tool(context),
        ]
    else:
        if tools_only:
            tools_to_persist = [build_revert_to_original_tool(context)]
        else:
            tools_to_persist = [
                build_python_env_code_run_tool(context),
                build_revert_to_original_tool(context),
            ]

    if tool_rag:
        obj_index = ObjectIndex.from_objects(
            tools,
            index_cls=VectorStoreIndex,
            embed_model=Settings.embed_model,
        )
        tool_retriever = obj_index.as_retriever(similarity_top_k=tool_rag_top_k)

        def retrieve_modded(self, str_or_query_bundle: str) -> list:
            query_bundle = QueryBundle(query_str=str_or_query_bundle)
            nodes = self._retriever.retrieve(query_bundle)
            for node_postprocessor in self._node_postprocessors:
                nodes = node_postprocessor.postprocess_nodes(
                    nodes, query_bundle=query_bundle
                )
            return [
                self._object_node_mapping.from_node(node.node) for node in nodes
            ] + tools_to_persist

        tool_retriever.retrieve = retrieve_modded.__get__(tool_retriever)
        if react:
            agent = ReActAgent.from_tools(
                llm=llm,
                tool_retriever=tool_retriever,
                verbose=True,
                system_prompt=system_prompt,
                memory=memory_obj,
                max_iterations=20,
            )
        else:
            if isinstance(llm, OpenAI) and not isinstance(llm, OpenAILike):
                agent = OpenAIAgent.from_tools(
                    llm=llm,
                    tool_retriever=tool_retriever,
                    verbose=True,
                    system_prompt=system_prompt,
                    memory=memory_obj,
                )
            else:
                agent = FunctionCallingAgent.from_tools(
                    llm=llm,
                    tool_retriever=tool_retriever,
                    verbose=True,
                    system_prompt=system_prompt,
                    memory=memory_obj,
                )
    else:
        if react:
            agent = ReActAgent.from_tools(
                llm=llm,
                tools=tools + tools_to_persist,
                verbose=True,
                system_prompt=system_prompt,
                memory=memory_obj,
                max_iterations=10,
            )
        else:
            if isinstance(llm, OpenAI) and not isinstance(llm, OpenAILike):
                agent = OpenAIAgent.from_tools(
                    llm=llm,
                    tools=tools + tools_to_persist,
                    verbose=True,
                    system_prompt=system_prompt,
                    memory=memory_obj,
                )
            else:
                agent = FunctionCallingAgent.from_tools(
                    llm=llm,
                    tools=tools + tools_to_persist,
                    verbose=True,
                    system_prompt=system_prompt,
                    memory=memory_obj,
                )
    return agent


class SingleAgent:
    def __init__(
        self,
        llm: FunctionCallingLLM,
        context: ToolingContext,
        react: bool,
        memory: Literal["buffer", "vector"] = "vector",
        memory_size: int = 3000,
        tool_rag: bool = True,
        tool_rag_top_k: int = 5,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        python_only: bool = False,
        tools_only: bool = False,
    ):
        """Initializes the SingleAgent object."""
        if not isinstance(llm, FunctionCallingLLM):
            raise ValueError("The provided LLM must be a FunctionCallingLLM.")

        self._context = context

        print_debug("Initializing SingleAgent")

        self._agent = build_agent(
            llm=llm,
            context=context,
            memory=memory,
            memory_size=memory_size,
            tool_rag=tool_rag,
            tool_rag_top_k=tool_rag_top_k,
            react=react,
            system_prompt=system_prompt,
            python_only=python_only,
            tools_only=tools_only,
        )

        self._build_agent_kwargs = {
            "llm": llm,
            "context": context,
            "memory": memory,
            "memory_size": memory_size,
            "tool_rag": tool_rag,
            "tool_rag_top_k": tool_rag_top_k,
            "system_prompt": system_prompt,
            "python_only": python_only,
            "tools_only": tools_only,
        }

        print_debug("SingleAgent instance initialized")

    def chat(self, message: str) -> str:
        """Interacts with the LLM to provide data analysis insights.

        Parameters
        ----------
        message : str
            The message to interact with the LLM.

        Returns
        -------
        str
            The response from the LLM.
        """

        def check_valid_output(output: str) -> bool:
            if (
                output is None
                or output.isspace()
                or output == "None"
                or output.startswith("<function=")
                or output.endswith("</function>")
            ):
                return False
            return True

        def recursive_validated_chat(
            message: str,
            max_retries: int = 5,
        ) -> str:
            for i in range(max_retries):
                try:
                    with suppress_logging():
                        output = str(self._agent.chat(message))
                    return output

                except Exception as e:
                    # if all retries are exhausted, return the default output
                    if i == max_retries:
                        return f"Error: {e}."
                    summary_str = str(build_dataset_summary_tool(self._context).call())
                    self._agent = build_agent(**self._build_agent_kwargs)
                    self._agent.memory.put(
                        ChatMessage.from_str(
                            content="Dataset summary: " + summary_str,
                            role=MessageRole.SYSTEM,
                        )
                    )
                    print_debug(
                        f"Agent reset after error: {e}. "
                        f"Dataset summary added to memory."
                    )
                    output = str(self._agent.chat(message))

                if not check_valid_output(output):
                    # if all retries are exhausted, return the default output
                    if i == max_retries:
                        return (
                            "I wasn't able to answer your question. "
                            "Could you please rephrase?"
                        )
                    self._agent.chat(
                        f"Your output, '{output}', doesn't answer the question. "
                        "Please try again. "
                        f"Here is the original question: '{message}'."
                    )
                    print_debug(f"Output '{output}' was invalid. Retrying.")

        return recursive_validated_chat(message)
