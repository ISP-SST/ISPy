#!/bin/env python
# -*- coding: utf-8 -*-

"""
Wavelength conversion vacuum-air python module

This python module is an example from the ISPy python library. It
illustrates how to wrap C-code with Cython. It's C source code was
extracted from the rh radiative transfer code and adapted to be
self-contained.
"""

try:
    from _vacuumtoair import *
except:
    raise ImportError("Could not load the `{0}' extension, check that it was properly compiled (or reinstall with the `--with-extensions' parameter)".format(__name__))
