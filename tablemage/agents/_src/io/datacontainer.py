import pandas as pd
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.schema import TextNode
import logging
from pathlib import Path

from ..options import options
from .._debug.logger import debug_log_path
from .... import Analyzer
from ....options import print_options


io_log_path = Path(__file__).resolve().parent / "_log" / "_log.log"
if io_log_path.exists():
    io_log_path.unlink()
    io_log_path.touch()
else:
    if not io_log_path.parent.exists():
        io_log_path.parent.mkdir(parents=True)
    io_log_path.touch()


def build_tablemage_analyzer(
    df: pd.DataFrame,
    df_test: pd.DataFrame | None = None,
    test_size: float = 0.2,
    split_seed: int = 42,
    name: str = "Dataset",
) -> Analyzer:
    """Builds a TableMage Analyzer for a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to build the Analyzer for.

    df_test : pd.DataFrame | None
        The test DataFrame to use for the Analyzer. Defaults to None.

    test_size : float
        The size of the test set. Defaults to 0.2.

    split_seed : int
        The seed to use for the train-test split. Defaults to 42.

    name : str
        The name of the dataset. Defaults to "Dataset".

    Returns
    -------
    tm.Analyzer
        The TableMage Analyzer.
    """
    print_options.reset_logger(logging.Logger("Blank"))
    debug_logger = logging.Logger("Analyzer Log")
    debug_filehandler = logging.FileHandler(debug_log_path)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    debug_filehandler.setFormatter(formatter)
    debug_logger.addHandler(debug_filehandler)

    io_filehandler = logging.FileHandler(io_log_path)
    io_filehandler.setFormatter(formatter)
    debug_logger.addHandler(io_filehandler)

    print_options.reset_secondary_logger(logger=debug_logger)

    analyzer = Analyzer(
        df,
        df_test=df_test,
        test_size=test_size,
        split_seed=split_seed,
        verbose=True,
        name=name,
    )
    return analyzer


class VariableInfo:
    def __init__(
        self,
        vars: list[str],
    ):
        """Initializes the VariableInfo object.

        Parameters
        ----------
        vars : list[str]
            The variables to provide information on.
        """
        self._in_memory_vector_index = VectorStoreIndex.from_documents([])
        self._vars_to_description = {var: None for var in vars}

    def set_description(
        self,
        var: str,
        description: str,
    ) -> None:
        """Sets the description for a variable.

        Parameters
        ----------
        var : str
            The variable to set the description for.

        description : str
            The description of the variable.
        """
        self._vars_to_description[var] = description
        self._in_memory_vector_index.insert_nodes(
            nodes=[
                TextNode(
                    text=f"{var}: {description}",
                    metadata={
                        "variable": var,
                        "description": description,
                    },
                )
            ]
        )

    def get_description(
        self,
        var: str,
    ) -> str:
        """Gets the description for a variable.

        Parameters
        ----------
        var : str
            The variable to get the description for.

        Returns
        -------
        str
            The description of the variable.
        """
        output = self._vars_to_description[var]
        if output is None:
            return ""


class DataContainer:
    def __init__(self):
        self.analyzer = None

    def set_analyzer(self, analyzer: Analyzer):
        """Sets the Analyzer for the DataContainer."""
        self.analyzer = analyzer
        self.df = self.analyzer.datahandler().df_all()
        self.variable_info = VariableInfo(vars=self.df.columns.to_list())

    def update_df(self):
        """Update the DataFrame based on the Analyzer's state."""
        self.df = self.analyzer.datahandler().df_all()
        self.variable_info = VariableInfo(vars=self.df.columns.to_list())
