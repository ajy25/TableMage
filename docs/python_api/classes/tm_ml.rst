Machine Learning Models :py:mod:`(tm.ml)`
=========================================


.. currentmodule:: tablemage.ml

The `tablemage.ml` module contains the machine learning models used by the 
:func:`.regress` and :func:`.classify` methods of the :class:`Analyzer` class. 
These models are designed to be used in a similar way to the models in the `sklearn` 
package, but with additional functionality for feature selection, 
hyperparameter optimization, and cross-validation.
        

:py:mod:`tm.ml.LinearR`
-----------------------

.. autoclass:: LinearR
    :members: 
        __init__, 
        specify_data, 
        fit, 
        sklearn_estimator, 
        sklearn_pipeline, 
        hyperparam_searcher, 
        fs_report, 
        is_cross_validated, 
        predictors, 
        feature_importance


:py:mod:`tm.ml.RobustLinearR`
-----------------------------

.. autoclass:: RobustLinearR
    :members: 
        __init__, 
        specify_data, 
        fit, 
        sklearn_estimator, 
        sklearn_pipeline, 
        hyperparam_searcher, 
        fs_report, 
        is_cross_validated, 
        predictors, 
        feature_importance


:py:mod:`tm.ml.TreesR`
----------------------

.. autoclass:: TreesR
    :members: 
        __init__, 
        specify_data, 
        fit, 
        sklearn_estimator, 
        sklearn_pipeline, 
        hyperparam_searcher, 
        fs_report, 
        is_cross_validated, 
        predictors, 
        feature_importance


:py:mod:`tm.ml.SVMR`
--------------------

.. autoclass:: SVMR
    :members: 
        __init__, 
        specify_data, 
        fit, 
        sklearn_estimator, 
        sklearn_pipeline, 
        hyperparam_searcher, 
        fs_report, 
        is_cross_validated, 
        predictors, 
        feature_importance


:py:mod:`tm.ml.MLPR`
--------------------

.. autoclass:: MLPR
    :members: 
        __init__, 
        specify_data, 
        fit, 
        sklearn_estimator, 
        sklearn_pipeline, 
        hyperparam_searcher, 
        fs_report, 
        is_cross_validated, 
        predictors, 
        feature_importance



:py:mod:`tm.ml.LinearC`
-----------------------

.. autoclass:: LinearC
    :members: 
        __init__, 
        specify_data, 
        fit, 
        sklearn_estimator, 
        sklearn_pipeline, 
        hyperparam_searcher, 
        fs_report, 
        is_cross_validated, 
        is_binary, 
        predictors, 
        feature_importance


:py:mod:`tm.ml.TreesC`
----------------------

.. autoclass:: TreesC
    :members: 
        __init__, 
        specify_data, 
        fit, 
        sklearn_estimator, 
        sklearn_pipeline, 
        hyperparam_searcher, 
        fs_report, 
        is_cross_validated, 
        is_binary, 
        predictors, 
        feature_importance


:py:mod:`tm.ml.SVMC`
--------------------

.. autoclass:: SVMC
    :members: 
        __init__, 
        specify_data, 
        fit, 
        sklearn_estimator, 
        sklearn_pipeline, 
        hyperparam_searcher, 
        fs_report, 
        is_cross_validated, 
        is_binary, 
        predictors, 
        feature_importance


:py:mod:`tm.ml.MLPC`
--------------------

.. autoclass:: MLPC
    :members: 
        __init__, 
        specify_data, 
        fit, 
        sklearn_estimator, 
        sklearn_pipeline, 
        hyperparam_searcher, 
        fs_report, 
        is_cross_validated, 
        is_binary, 
        predictors, 
        feature_importance
