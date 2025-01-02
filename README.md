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

We recommend installing TableMage in a virtual environment.

To install TableMage:
```
git clone https://github.com/ajy25/TableMage.git
cd TableMage
pip install .
```

For notes regarding dependencies, please refer to `./dev_notes/dependencies.md`. 
TableMage supports Python versions 3.10 through 3.12.

**Note for MacOS users:** You might run into an error involving XGBoost, one of TableMage's dependencies, when using TableMage for the first time.
To resolve this error, you'll need to install libomp: `brew install libomp`. This requries [Homebrew](https://brew.sh/).

## Quick start (low-code)

You'll likely use TableMage for machine learning model benchmarking. Here's how to do it.

```python
import tablemage as tm
import pandas as pd
import joblib

# load table (we'll assume 'y' is the numeric variable to predict)
df = ...

# initialize an Analyzer object
analyzer = tm.Analyzer(df, test_size=0.2)

# preprocess data
analyzer.dropna(
    include_vars=['y']
).impute(
    exclude_vars=['y']
).scale(
    exclude_vars=['y']
)

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

Check out the `./demo` directory for detailed examples and walkthroughs of other functionality, 
including exploratory data analysis, regression analysis, and causal inference.


## Quick start (no-code)

First, install the required additional dependencies.
```
pip install '.[agents]'
```

Next, add your API key. You only need to do this once; your API key will be written to a local `.env` file.
```python
import tablemage as tm
tm.use_agents()                                             # import the agents module
tm.agents.set_key("openai", "add-your-api-key-here")        # set API key
```

You can open up a chat user interface by running the following code and clicking on the URL that appears in the terminal.
Your conversation with the AI agent appears on the left, while the AI agent's analyses (figures made, tables produced, TableMage commands used) appear on the right.

```python
import tablemage as tm
tm.use_agents()
tm.agents.App().run()
```

Or, you can chat with the AI agent directly in Python:

```python
import pandas as pd
import tablemage as tm
tm.use_agents()

# load table
df = ...

# initialize a ConversationalAgent object
agent = tm.agents.ConversationalAgent(df, test_size=0.2)

# chat with the agent
print(agent.chat("Compute the summary statistics for the numeric variables."))
```

## Notes

TableMage is under active development.
