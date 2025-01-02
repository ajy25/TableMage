"""
TableMage
---------
Python package for low-code/no-code data science on tabular data.
"""

from ._src.analyzer import Analyzer
from ._src.display.print_utils import print_wrapped
from . import ml
from . import options
from . import fs


def use_agents():
    """Import the agents module."""
    global __all__
    global agents
    # try to import the agents module
    try:
        from . import agents

        if "agents" in locals():
            __all__.append("agents")

        print_wrapped(
            text="The `agents` module has been imported.", type="UPDATE", level="INFO"
        )
    except Exception:
        print_wrapped(
            text="Could not import the `agents` module.", type="WARNING", level="INFO"
        )


__all__ = ["Analyzer", "ml", "options", "fs"]

__version__ = "0.1.0a1"
__author__ = "Andrew J. Yang"
