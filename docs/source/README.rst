ISPy (https://github.com/ISP-SST/ISPy) is a Python library of commonly used tools at the Institute for Solar Physics (https://www.isf.astro.su.se/) at the Stockholm University (https://www.su.se), covering SST data processing and alignment, radiative transfer calculations and inversion pre- and post-processing, among others. Read this documentation (https://ISP-SST.github.io/ISPy/) for getting details on how to use the code.

Getting Started
---------------

Since ISPy is still in a development phase, the only possible way of
installing the code is via compilation of the sources.
The sources can be installed by cloning this repository and installing it:

::
    
    git clone https://github.com/ISP-SST/ISPy
    python setup.py install

Do not forget to pull if there is a new version using the `repository <https://github.com/ISP-SST/ISPy>`_ and recompiling the code by typing the following from the location of the sources:

::
    
    git pull
    python setup.py install
    

Requirements
------------
ISPy depends on the following external packages, that should be straightforward to install using conda:

* ``sunpy``
* ``numpy``
* ``astropy``
* ``scipy``
* ``astropy``
* ``matplotlib``
* ``ipdb``

If a Python module is needed for ANA f0 files you can clone and install it from this repository: https://github.com/cdiazbas/pyana