Getting Started
===============

Installation
------------

TableMage officially supports Python versions 3.10 through 3.12.
We recommend installing **TableMage** in a new virtual environment to avoid dependency conflicts.
To install TableMage, follow these steps:

.. code-block:: bash

    git clone https://github.com/ajy25/TableMage.git
    cd TableMage
    pip install .

.. note::

    **For MacOS users:**  
    You might encounter an error involving XGBoost, one of TableMage's dependencies, when using TableMage for the first time.  
    To resolve this issue, install `libomp` by running:

    .. code-block:: bash

        brew install libomp

    This requires `Homebrew`. For more information, visit the `Homebrew website <https://brew.sh/>`_.


Demo
----
Check out the following demo to see **TableMage** in action:

.. toctree::
   :maxdepth: 1
   :caption: Demo

   demo.ipynb