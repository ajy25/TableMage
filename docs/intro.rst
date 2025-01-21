Introduction
============

.. currentmodule:: tablemage

Domain expertise is extremely valuable in data science and quantitative research.
However, experts in fields such as medicine and healthcare often lack the 
necessary statistical and computational experience to analyze data on their own.

TableMage simplifies data science, making it
accessible to researchers with little to no quantitative background. 
TableMage empowers domain experts to perform their own data
analysis and machine learning modeling without needing to learn the intricacies 
of data science (Python, pandas, scikit-learn, ML theory). 
TableMage's Python API is designed to be intuitive and easy to use, with 
low-code functions that abstract away the complexity of data science. 

Below, we list the steps of a typical data science workflow and describe
how TableMage can help at each step. 

.. rubric:: Data Loading and Cleaning

Data must be extracted and cleaned prior to analysis. 
Unfortunately, TableMage does not handle data loading and cleaning. 
TableMage works with wide format tabular data in the form of a pandas DataFrame.
See :class:`tablemage.Analyzer` for details.


.. rubric:: Data Exploration

New datasets need to be explored prior to further processing and modeling.
Data scientists working with Python typically use pandas and matplotlib/seaborn
to explore data. 

TableMage enhances the data exploration process by providing a 
comprehensive report object that includes the following:

1. Categorical and numeric variable summary statistics
2. Low-code statistical testing
3. Low-code visualizations

See :meth:`tablemage.Analyzer.eda` for details.


.. rubric:: Data Preprocessing

Before modeling, data must be preprocessed (e.g., missing values imputed, 
categorical variables encoded, etc.). 

TableMage streamlines the data preprocessing process, reducing
several minutes of documentation reading and 10+ lines of code down to zero minutes of 
documentation reading and one line of code. 

See :meth:`tablemage.Analyzer.impute`, 
:meth:`tablemage.Analyzer.scale`, 
:meth:`tablemage.Analyzer.drop_highly_missing_vars`, 
:meth:`tablemage.Analyzer.dropna`,
and :meth:`tablemage.Analyzer.onehot` for details. 

.. note::
    Categorical variables are automatically one-hot encoded in the modeling process
    if you do not one-hot encode them beforehand. Also, observations with missing values 
    are automatically dropped in the modeling process. 


.. rubric:: Statistical Modeling

Statistical modeling is sometimes preferred over machine learning modeling
because it provides interpretable results.

TableMage improves the statistical modeling process by providing low-code 
functions for linear regression analysis. 

See 
:meth:`tablemage.Analyzer.ols` and :meth:`tablemage.Analyzer.logit` for details.


.. rubric:: Machine Learning Modeling

Machine learning modeling is often preferred over statistical modeling
because it can handle more complex relationships in the data.

TableMage simplifies the machine learning modeling process by providing
low-code functions for classification and regression models. 

See
:meth:`tablemage.Analyzer.classify` and :meth:`tablemage.Analyzer.regress` for 
details. The different models available are listed in 
:doc:`../python_api/classes/tm_ml`.





