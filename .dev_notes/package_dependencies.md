# TableMage dependencies

All dependencies are listed in the `pyproject.toml` file at the root.

### Boruta_Py

Boruta_Py, a scikit-learn style implementation of the popular Boruta feature selection algorithm invented by Miron B. Kursa, was developed by Daniel Homola in 2016. The package was released under the BSD 3-Clause license. The Boruta_Py repository lives on [GitHub](https://github.com/scikit-learn-contrib/boruta_py).

At time of development, due to Boruta_Py's incompatibility with NumPy versions 1.24.3 and later (specifically, ``np.int`` and similar data types were deprecated in these NumPy versions), we decided to maintain a copy of the file in the TableMage repository. The Boruta_Py license has been inserted at the top of the `boruta_py.py` file.