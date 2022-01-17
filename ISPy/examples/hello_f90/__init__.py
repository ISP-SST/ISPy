#!/bin/env python
# -*- coding: utf-8 -*-

"""
F2py wrapper on Fortran 90 hello-world module

This python module is an example from the ISPy python library. It
illustrates how to wrap Fortran 90 code with F2py.
"""

try:
    from _helloworld import *
except:
    raise ImportError("Could not load the `{0}' extension, check that it was properly compiled (or reinstall with the `--with-extensions' parameter)".format(__name__))

hello = helloworld.hello
