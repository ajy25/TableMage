import pandas as pd
from typing import Literal
from .._src import (
    build_tablemage_analyzer,
    StorageManager,
    DataContainer,
    CanvasQueue,
    ToolingContext,
    print_debug,
)
from ..._src.display.print_utils import suppress_logging
from .._src.agents_src.single_agent import SingleAgent
from .._src.agents_src.prompt.single_agent_system_prompt import DEFAULT_SYSTEM_PROMPT
from .._src.options import options


class ConversationalAgent:
    """Class for interacting with the LLMs for data analysis on tabular data."""

    def __init__(
        self,
        df: pd.DataFrame,
        df_test: pd.DataFrame | None = None,
        test_size: float = 0.2,
        split_seed: int = 42,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        react: bool = False,
        memory_type: Literal["buffer", "vector"] = "vector",
        memory_size: int = 3000,
        tool_rag: bool = True,
        tool_rag_top_k: int = 5,
        python_only: bool = False,
        tools_only: bool = False,
    ):
        """Initializes the ConversationalAgent object.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to build the Analyzer for.

        df_test : pd.DataFrame | None
            The test DataFrame to use for the Analyzer. Defaults to None.

        test_size : float
            The size of the test set. Defaults to 0.2.

        split_seed : int
            The seed to use for the train-test split. Default is 42.

        system_prompt : str
            The system prompt to use for the LLM. Default is provided.

        react: bool
            If True, the agent will employ the ReAct framework. Default is False.

        memory_type : Literal["buffer", "vector"]
            The type of memory to use. Default is "vector".

        memory_size : int
            The size of the memory to use. Token limit synonym. Default is 3000.

        tool_rag : bool
            If True, the RAG-based tooling is used. Default is True.

        tool_rag_top_k : int
            The top-k value to use for the RAG-based tooling. Default is 5.

        python_only : bool
            If True, only the Python environment is provided.
            Default is False.

        tools_only : bool
            If True, only the tools are provided. Otherwise, the
            Python environment is also provided, ignoring RAG. Default is False.
        """
        self._data_container = DataContainer()
        self._data_container.set_analyzer(
            build_tablemage_analyzer(
                df,
                df_test=df_test,
                test_size=test_size,
                split_seed=split_seed,
            )
        )
        print_debug(
            "Data container initialized with the Analyzer built from the "
            "provided DataFrame."
        )
        self._vectorstore_manager = StorageManager(multimodal=False, vectorstore=False)
        self._canvas_queue = CanvasQueue()
        self._context = ToolingContext(
            data_container=self._data_container,
            storage_manager=self._vectorstore_manager,
            canvas_queue=self._canvas_queue,
        )
        print_debug("IO initialized.")
        print_debug("Initializing the Agent.")
        self._single_agent = SingleAgent(
            llm=options.llm_build_function(),
            context=self._context,
            react=react,
            memory=memory_type,
            memory_size=memory_size,
            tool_rag_top_k=tool_rag_top_k,
            tool_rag=tool_rag,
            system_prompt=system_prompt,
            python_only=python_only,
            tools_only=tools_only,
        )
        print_debug(
            f"Agent initialized. Agent type: {self._single_agent.__class__.__name__}"
        )

    def chat(self, message: str) -> str:
        """Interacts with the LLM to provide data analysis insights.

        Parameters
        ----------
        message : str
            The message to send to the LLM.

        Returns
        -------
        str
            The response from the LLM.
        """
        return self._single_agent.chat(message)
