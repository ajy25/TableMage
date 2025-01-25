Machine Learning Models :py:mod:`(tm.ml)`
=========================================


.. currentmodule:: tablemage

The `tablemage.ml` module contains the machine learning models used by the 
:func:`tablemage.Analyzer.regress` and :func:`tablemage.Analyzer.classify` methods of the :class:`tablemage.Analyzer` class. 
These models are designed to be used in a similar way to the models in the `scikit-learn` 
package, but with additional functionality for feature selection, 
hyperparameter optimization, and cross-validation.
        

:py:mod:`tm.ml.LinearR`
-----------------------

.. autoclass:: tablemage.ml.LinearR
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

.. autoclass:: tablemage.ml.RobustLinearR
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

.. autoclass:: tablemage.ml.TreesR
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

.. autoclass:: tablemage.ml.SVMR
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

.. autoclass:: tablemage.ml.MLPR
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

.. autoclass:: tablemage.ml.LinearC
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

.. autoclass:: tablemage.ml.TreesC
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

.. autoclass:: tablemage.ml.SVMC
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

.. autoclass:: tablemage.ml.MLPC
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
