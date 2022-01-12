.. ISPy documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ISPy's documentation!
================================

ISPy (https://github.com/ISP-SST/ISPy) is a Python library of commonly used tools at the Institute for Solar Physics (https://www.isf.astro.su.se/) at the Stockholm University (https://www.su.se), covering SST data processing and alignment, radiative transfer calculations and inversion pre- and post-processing, among others. Read this documentation (https://ISP-SST.github.io/ISPy/) for getting details on how to use the code.

Getting Started
---------------

ISPy can be installed via this repository:
::
   git clone https://github.com/ISP-SST/ISPy
   python setup.py install

or via pip by using the following command:
::
   pip install git+https://github.com/ISP-SST/ISPy



Requirements
------------
ISPy depends on the following external packages, that should be straightforward to install using conda: ``numpy``, ``astropy``, ``scipy``, ``astropy``, ``matplotlib``, ``Cython``, ``h5py``, ``ipdb``, ``tensorflow``, ``keras``, ``tqdm``. If a Python module is needed for ANA f0 files you can clone and install it from this repository: https://github.com/cdiazbas/pyana


.. toctree::
   :maxdepth: 3

   .. README

Documentation
=============

Here you will find an index and the list of all the modules with their descriptions. You can also search for words/keywords using the upper left panel.

.. toctree::
   :maxdepth: 3

   ISPy

.. Documentation
.. =========================

.. Here you will find an index and the list of all the modules with their descriptions. You can also search for words/keywords using the upper left panel.

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

