Reports :py:mod:`(tm._reports)`
===============================

Report objects are outputted by the :meth:`tablemage.Analyzer.eda`, 
:meth:`tablemage.Analyzer.ols`, :meth:`tablemage.Analyzer.logit`, 
:meth:`tablemage.Analyzer.regress`, 
and :meth:`tablemage.Analyzer.classify` methods of the :class:`tablemage.Analyzer` class. 
They may contain information about model performance, feature importance, or other
relevant statistics. They also have methods for plotting relevant diagnostic figures.


.. currentmodule:: tablemage._reports


:py:mod:`tm._reports.MLClassificationReport`
--------------------------------------------

.. autoclass:: tablemage._reports.MLClassificationReport
    :members:
        model, 
        metrics, 
        cv_metrics, 
        fs_report, 
        plot_confusion_matrix, 
        plot_roc_curves,
        plot_roc_curve, 
        metrics_by_class, 
        cv_metrics_by_class, 
        feature_importance,
        is_binary


:py:mod:`tm._reports.MLRegressionReport`
----------------------------------------

.. autoclass:: tablemage._reports.MLRegressionReport
    :members:
        model, 
        metrics, 
        cv_metrics, 
        fs_report, 
        plot_obs_vs_pred, 
        feature_importance

:py:mod:`tm._reports.OLSReport`
-------------------------------

.. autoclass:: tablemage._reports.OLSReport
    :members:
        model,
        metrics, 
        step, 
        test_lr,
        test_partialf, 
        statsmodels_summary, 
        plot_obs_vs_pred, 
        plot_residuals_vs_fitted, 
        plot_residuals_vs_var, 
        plot_residuals_hist,
        plot_scale_location, 
        plot_residuals_vs_leverage, 
        plot_qq, 
        plot_diagnostics, 
        set_outlier_threshold, 
        get_outlier_indices,
        coefs


:py:mod:`tm._reports.EDAReport`
-------------------------------

.. autoclass:: tablemage._reports.EDAReport
    :members:
        tabulate_correlation_comparison,
        tabulate_correlation_matrix,
        tabulate_tableone,
        plot_pairs,
        plot,
        plot_pca,
        plot_correlation_heatmap,
        test_equal_means,
        test_normality,
        test_categorical_independence,
        chi2,
        anova,
        ttest,
        numeric_vars,
        numeric_stats,
        categorical_vars,
        categorical_stats,
        value_counts
        

:py:mod:`tm._reports.VotingSelectionReport`
-------------------------------------------

.. autoclass:: tablemage._reports.VotingSelectionReport
    :members:
        top_features, all_features, votes

:py:mod:`tm._reports.StatisticalTestReport`
-------------------------------------------

.. autoclass:: tablemage._reports.StatisticalTestReport
    :members:
        pval, 
        statistic
        

:py:mod:`tm._reports.CausalReport`
----------------------------------

.. autoclass:: tablemage._reports.CausalReport
    :members:
        effect,
        se,
        n_units,
        pval



