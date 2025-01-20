import matplotlib.pyplot as plt
import pandas as pd
from json import dumps

from ..._src import DataContainer, StorageManager, CanvasQueue
from .._debug.logger import print_debug


class ToolCall:
    """Class for storing tool calls."""

    def __init__(self, tool_fn_name: str, tool_fn_args: dict):
        """Initializes the ToolCall object.

        Parameters
        ----------
        tool_name : str
            The name of the tool.

        tool_args : dict
            The arguments of the tool.
        """
        self.tool_name = tool_fn_name
        self.tool_args = tool_fn_args

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolCall):
            return False
        return self.tool_name == other.tool_name and self.tool_args == other.tool_args

    def __str__(self) -> str:
        return f"{self.tool_name}({self.tool_args})"


class ToolingContext:
    def __init__(
        self,
        data_container: DataContainer,
        storage_manager: StorageManager,
        canvas_queue: CanvasQueue,
    ):
        """Initializes the ToolingContext object.

        Parameters
        ----------
        data_container : DataContainer
            The DataContainer object to use retrieving data.

        storage_manager : StorageManager
            The StorageManager object to use for storing information and analysis.

        canvas_queue : CanvasQueue
            The CanvasQueue object to use for storing images, tables, etc. for UI use.
        """
        self._data_container = data_container
        self._storage_manager = storage_manager
        self._canvas_queue = canvas_queue
        self._toolcalls = []

    def add_figure(
        self,
        fig: plt.Figure,
        text_description: str,
        augment_text_description: bool = True,
    ) -> str:
        """Adds a figure.

        Parameters
        ----------
        fig : plt.Figure
            Figure to add to the vector index.

        text_description : str
            Description of the figure.

        augment_text_description : bool
            Whether to augment the text description with a vision model,
            by default True

        Returns
        -------
        str
            Description of the figure.
        """
        descr, path = self._storage_manager.add_figure(
            fig=fig,
            text_description=text_description,
            augment_text_description=augment_text_description,
        )
        self._canvas_queue.push_figure(path)
        return descr

    def add_str(self, text: str) -> str:
        """Adds a string.

        Parameters
        ----------
        text : str
            Text to add to the vector index.

        Returns
        -------
        str
            The input text, verbatim.
        """
        return self._storage_manager.add_str(text)

    def add_table(self, table: pd.DataFrame, add_to_vectorstore: bool = True) -> str:
        """Adds a pandas DataFrame.

        Parameters
        ----------
        table : dict
            Table to add to the canvas (and optionally, the vector index).

        add_to_vectorstore : bool
            Whether to add the table to the vector index, by default True.
            May be set to False if a custom dict including the DataFrame
            is to be added to the vector store (e.g. use add_dict instead).

        Returns
        -------
        str
            The input table in json string format.
        """
        print_debug(f"Adding table to storage.")
        strres, path = self._storage_manager.add_table(table, add_to_vectorstore)
        self._canvas_queue.push_table(path)
        print_debug(f"Added table to {path}: {strres}")
        return strres

    def add_dict(self, dictionary: dict, description: str | None = None) -> str:
        """Adds a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary to add to the vector index.

        description : str, optional
            Description of the dictionary, by default None

        Returns
        -------
        str
            The input dictionary in json string format.
        """
        print_debug(f"Adding dictionary to storage.")
        str_dict = dumps(dictionary)
        if description is not None:
            str_dict = description + "\n\n" + str_dict
            print_debug(f"Added dictionary with description: {description}")
        else:
            print_debug(
                f"Added dictionary without description. "
                f"First 20 characters: {str_dict[:20]}."
            )
        return self._storage_manager.add_str(str_dict)

    def add_thought(self, thought: str) -> str:
        """Adds a thought to the canvas queue.

        Parameters
        ----------
        thought : str
            Thought to add to the vector index.

        Returns
        -------
        str
            The input thought, verbatim.
        """
        return self._canvas_queue.push_thought(thought)

    def add_code(self, code: str) -> str:
        """Adds code to the canvas queue.

        Parameters
        ----------
        code : str
            Code to add to the vector index.

        Returns
        -------
        str
            The input code, verbatim.
        """
        return self._canvas_queue.push_code(code)

    def add_toolcall(self, toolcall: ToolCall) -> None:
        """Adds a tool call to the context.

        Parameters
        ----------
        toolcall : ToolCall
            The tool call to add.
        """
        self._toolcalls.append(toolcall)

    def is_repeat_toolcall(self, toolcall: ToolCall) -> bool:
        """Checks if a tool call is a repeat of the most recent tool call.

        Parameters
        ----------
        toolcall : ToolCall
            The tool call to check.

        Returns
        -------
        bool
            True if the tool call is a repeat, False otherwise.
        """
        if len(self._toolcalls) < 3:
            return False
        return (
            toolcall == self._toolcalls[-1]
            and toolcall == self._toolcalls[-2]
            and toolcall == self._toolcalls[-3]
        )

    @property
    def data_container(self) -> DataContainer:
        return self._data_container

    @property
    def storage_manager(self) -> StorageManager:
        return self._storage_manager

    @property
    def canvas_queue(self) -> CanvasQueue:
        return self._canvas_queue
