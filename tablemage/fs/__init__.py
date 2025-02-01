"""
TableMage Feature Selection Module
-----------------------------------
This module cotnains classes that provide feature selection functionality. \
All classes are compatible with the `tm.regress()`, `tm.classify()`, and \
`tm.select_features()` functions.

Classes, Objects, and Functions
--------------------------------
KBestFSR : A class that provides feature selection using the k-best method for \
regression tasks.

LassoFSR : A class that provides feature selection using lasso regression for \
regression tasks.

BorutaFSR : A class that provides feature selection using the Boruta algorithm \
for regression tasks.

KBestFSC : A class that provides feature selection using the k-best method for \
classification tasks.

LassoFSC : A class that provides feature selection using lasso logistic regression for \
classification tasks.

BorutaFSC : A class that provides feature selection using the Boruta algorithm \
for classification tasks.
"""

from .._src.feature_selection import (
    KBestFSR,
    LassoFSR,
    BorutaFSR,
    KBestFSC,
    LassoFSC,
    BorutaFSC,
)

__all__ = ["KBestFSR", "LassoFSR", "BorutaFSR", "KBestFSC", "LassoFSC", "BorutaFSC"]
