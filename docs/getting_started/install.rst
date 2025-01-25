Package Installation
====================

We recommend installing **TableMage** in a virtual environment to avoid dependency conflicts.

Installation
------------

To install TableMage, follow these steps:

.. code-block:: bash

    git clone https://github.com/ajy25/TableMage.git
    cd TableMage
    pip install .

Supported Python Versions
-------------------------

TableMage officially supports Python versions 3.10 through 3.12.

.. note::

    **For MacOS users:**  
    You might encounter an error involving XGBoost, one of TableMage's dependencies, when using TableMage for the first time.  
    To resolve this issue, install `libomp` by running:

    .. code-block:: bash

        brew install libomp

    This requires `Homebrew`. For more information, visit the `Homebrew website <https://brew.sh/>`_.