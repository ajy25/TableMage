"""
TableMage
---------
Python package for low-code/no-code data science on tabular data.
"""

from ._src.analyzer import Analyzer
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
    except Exception:
        pass


__all__ = ["Analyzer", "ml", "options", "fs"]

__version__ = "0.1.0a1"
__author__ = "Andrew J. Yang"
