# TableMage dependencies

## tablemage

Dependencies that can be found on PyPI are included below.

```python
numpy>=1.24.3
pandas>=2.2.2
scipy>=1.1.3
scikit-learn>=1.5.0
matplotlib>=3.8.0
seaborn>=0.13.2
xgboost>=2.0.3
statsmodels>=0.14.0
adjustText>=1.1.1
optuna>=3.6.1
optuna-integration>=3.6.0
```

### Boruta_Py

Boruta_Py, a scikit-learn style implementation of the popular Boruta feature selection algorithm invented by Miron B. Kursa, was developed by Daniel Homola in 2016. The package was released under the BSD-3-Clause license. The Boruta_Py repository lives on [GitHub](https://github.com/scikit-learn-contrib/boruta_py).

Due to Boruta_Py's incompatibility with NumPy versions 1.24.3 and later (specifically, ``np.int`` and similar data types were deprecated in these NumPy versions), we maintain a copy of the file in the TableMage repository. The Boruta_Py license has been inserted at the top of the `boruta_py.py` file.
