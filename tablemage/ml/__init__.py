"""
TableMage Machine Learning Models
---------------------------------
This module contains machine learning models for use with TableMage. \
All classes are compatible with the `tm.regress()` and `tm.classify()`.

Classes, Objects, and Functions
-------------------------------
LinearR : Linear regression model.

RobustLinearR : Robust linear regression model.

TreesR : Decision tree regression model.

MLPR : Multi-layer perceptron regression model.

SVMR : Support vector machine regression model.

CustomR : Custom regression model.

LinearC : Linear classification model.

TreesC : Decision tree classification model.

MLPC : Multi-layer perceptron classification model.

SVMC : Support vector machine classification model.

CustomC : Custom classification model.

KMeansClust : KMeans clustering model.

GMMClust : Gaussian mixture model clustering model.
"""

from .._src.ml.predict.regression import (
    LinearR,
    RobustLinearR,
    TreesR,
    MLPR,
    SVMR,
    CustomR,
)
from .._src.ml.predict.classification import (
    LinearC,
    TreesC,
    MLPC,
    SVMC,
    CustomC,
)

from .._src.ml.cluster import (
    KMeansClust,
    GMMClust,
)


__all__ = [
    "LinearR",
    "RobustLinearR",
    "TreesR",
    "MLPR",
    "SVMR",
    "CustomR",
    "LinearC",
    "TreesC",
    "CustomC",
    "MLPC",
    "SVMC",
    "GMMClust",
    "KMeansClust",
]
