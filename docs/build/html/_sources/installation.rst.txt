Installation
============

Conda virtual environments are preferred for machine learning projects where dependency version numbers are very important, and might vary from one package to another.
If your machine has a GPU, you can additionally install GPU support


# Install Python Dependencies


Setup Environment:

.. code-block:: bash

   pip3 install -e .

For easy install on WSL and Ubuntu use:

.. code-block:: bash

   pip install -r requirements.txt

Tested to work on Windows 11 WSL with the following:

* Python Version 3.12
* Conda Version 23.7.4

