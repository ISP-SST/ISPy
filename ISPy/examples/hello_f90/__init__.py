#!/bin/env python
# -*- coding: utf-8 -*-

"""
F2py wrapper on Fortran 90 hello-world module

This python module is an example from the ISPy python library. It
illustrates how to wrap Fortran 90 code with F2py.
"""

from _helloworld import *

hello = helloworld.hello
