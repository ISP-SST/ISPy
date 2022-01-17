#!/bin/env
# -*- coding: utf-8 -*-

"""
Cython wrapper on hello-world C++ module

This hello-world python module illustrates how to interface C++ classes with
Cython for the ISPy package. For interfacing functions involving arrays, see
the `airtovacuum' example module (in C) that applies similarly to C++.

This module implements the function `hello', a simple hello-world, and the
class Hello, with its printMessage function.
"""

try:
    from _helloworld import *
except:
    raise ImportError("Could not load the `{0}' extension, check that it was properly compiled (or reinstall with the `--with-extensions' parameter)".format(__name__))
