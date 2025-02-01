"""
TableMage
=========
A Python package for rapid low-code clinical data science. \
Contains tools for exploratory data analysis, regression analysis, and machine learning.

Classes, Objects, and Functions
-------------------------------
Analyzer : Main class for TableMage. Used to analyze data and train/evaluate models.

use_agents : Function to import the agents module.

Submodules
----------
ml : Module containing machine learning models.

fs : Module containing feature selectors.

agents : Module containing AI agents and helper functions for conversational data analysis.

options : Module containing print and plot option setters for Analyzer methods.
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
            text="The 'tablemage.agents' module has been imported.",
            type="UPDATE",
            level="INFO",
        )
    except Exception as e:
        print_wrapped(
            text="Could not import the 'tablemage.agents' module. "
            "Exception: {}".format(e),
            type="WARNING",
            level="INFO",
        )


__all__ = ["Analyzer", "ml", "options", "fs", "use_agents"]

__version__ = "0.1.0a1"
__author__ = "Andrew J. Yang"
