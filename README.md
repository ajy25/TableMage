# TableMage

![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Tests Passing](https://github.com/ajy25/TableMage/actions/workflows/test.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/tablemage/badge/?version=latest)](https://tablemage.readthedocs.io/en/latest/?badge=latest)



TableMage is a Python package for low-code/no-code data science.
TableMage can help you quickly explore tabular datasets, 
easily perform regression analyses, 
and effortlessly compute performance metrics for your favorite machine learning models.


## Installation and dependencies

To install TableMage:
```
git clone https://github.com/ajy25/TableMage.git
cd TableMage
pip install .
```

To uninstall TableMage:
```
pip uninstall TableMage
```

TableMage is built with the standard Python data science stack (scikit-learn, scipy, statsmodels).
For additional notes regarding dependencies, check out `./dev_notes/dependencies.md`. 
TableMage requires Python version 3.10 or later.

## Quick start (low-code)

You'll probably use TableMage for ML model benchmarking. Here's how to do it.

```python
import tablemage as tm
import pandas as pd
import joblib

# load table (we'll assume 'y' is the numeric variable to predict)
df = ...

# initialize an Analyzer object
analyzer = tm.Analyzer(df, test_size=0.2)

# preprocess data
analyzer.dropna(['y']).impute().scale()

# train regressors (hyperparameter tuning is preset and automatic)
reg_report = analyzer.regress(
    models=[
        tm.ml.LinearR('l2'),
        tm.ml.TreesR('random_forest'),
        tm.ml.TreesR('xgboost'),
    ],
    target='y',
    feature_selectors=[
        tm.fs.BorutaFSR()   # select features
    ]
)

# compare model performance
print(reg_report.metrics('test'))

# predict on new data
new_df = ...
y_pred = reg_report.model('LinearR(l2)').predict(new_df)

# save model as sklearn pipeline
joblib.dump(reg_report.model('LinearR(l2)'), 'l2_pipeline.joblib')
```

Check out the `./demo` directory for detailed examples and discussion of other functionality.




## Quick start (no-code)

You can open up a chat user interface by running the following code and clicking on the URL that appears in the terminal.
Your conversation with the AI agent appears on the left, while the AI agent's analyses (figures made, tables produced, TableMage commands used) appear on the right.

```python
import tablemage as tm
tm.mage.App().run()
```

Or, you can chat with the AI agent directly in Python:

```python
import tablemage as tm
import pandas as pd

# load table
df = ...

# initialize a Mage object
mage = tm.mage.Mage(df, test_size=0.2)

# chat with the Mage
print(mage.chat("Compute the summary statistics for the numeric variables."))
```

## Notes

TableMage is under active development.

### Motivation: low-code/no-code data science for clinical research

TableMage provides a low-code Python API that exponentially accelerates data science and machine learning by seamlessly connecting the data exploration and processing steps to the modeling steps. TableMage offers the following:
1. **Preprocess-as-you-explore functionality.** TableMage remembers data transformations and automatically preprocesses your train, validation, and test datasets when you later fit and evaluate models. 
2. **Automatic hyperparameter optimization and feature selection.** TableMage can automatically select features and identify optimal hyperparameters for you. All TableMage ML models come with preset hyperparameter search methods. 
3. **Flexibility.** Though TableMage provides many out-of-the-box models with default hyperparameter search spaces, it also supports custom estimators and pipelines. Any scikit-learn `BaseEstimator`/`Pipeline`-like object with fit and predict methods can be used. 
4. **Linear regression.** TableMage contains numerous methods to support statsmodels' classical linear statistical models, including diagnostic plots, stepwise feature selection, and statistical tests, enabling you to seamlessly switch between linear statistical modeling and ML modeling.
5. **Exploratory data analysis, causal inference, clustering, and more.** TableMage provides low-code tools for other areas of data science, not just regression and classification.
5. **GenAI integration.**  TableMage comes with AI agents equipped with its low-code tools, allowing you to chat with your data.

