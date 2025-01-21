"""
Options
-------
The options module contains functions for setting and displaying options for
printing and plotting.

Objects
-------
print_options : Object to set options for printing.
plot_options : Object to set options for plotting.
"""

from .._src.display.print_options import print_options
from .._src.display.plot_options import plot_options

__all__ = [
    "print_options",
    "plot_options",
]
