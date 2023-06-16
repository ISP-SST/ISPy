#!/bin/env python
# -*- cofing: utf-8 -*-


"""
Execute like this:

$ python bin/generate_module_list.py
modules = [
    'ISPy.img',
    'ISPy.inv',
    'ISPy.io',
    'ISPy.rt',
    'ISPy.sim',
    'ISPy.spec',
    'ISPy.util',
    'ISPy.vis',
...
]

"""

from glob import glob
from os.path import dirname, join, isfile, sep
from pathlib import Path

def get_paths(base="ISPy", level=15):
    """
    Generates a set of paths for modules searching.

    Examples
    ========

    >>> get_paths('ISPy', 2)
    ['ISPy/__init__.py', 'ISPy/*/__init__.py', 'ISPy/*/*/__init__.py']
    >>> get_paths('ISPy', 6)
    ['ISPy/__init__.py', 'ISPy/*/__init__.py', 'ISPy/*/*/__init__.py',
    'ISPy/*/*/*/__init__.py', 'ISPy/*/*/*/*/__init__.py',
    'ISPy/*/*/*/*/*/__init__.py', 'ISPy/*/*/*/*/*/*/__init__.py']

    """
    wildcards = ["/"]
    for i in range(level):
        wildcards.append(wildcards[-1] + "*/")
    p = [base + x + "__init__.py" for x in wildcards]
    return p

def generate_module_list(wdir, with_cython=False, with_ext=False):
    """
    Generates a list of all available modules

    When with_cython and/or with_ext are provided, the list restricts
    to those modules that require cython and/or the compilations of an
    extension.
    """
    g = []
    preflen = len(str(Path(__file__).parents[1].absolute()))
    if str(Path(__file__).parents[1].absolute()) != wdir[:preflen]:
        raise RuntimeError('Working directory outside of ISPy')
    for x in get_paths(wdir[preflen+1:] or 'ISPy'):
        for y in glob(str(Path(__file__).parents[1].absolute().joinpath(x))):
            ok_cython = True
            ok_ext = True
            if len(glob(join(dirname(y),'*.pyx'))) == 0:
                ok_cython = not with_cython
            if not isfile(join(dirname(y),'__extensions__.ispy')):
                ok_ext = not with_ext
            if ok_cython and ok_ext:
                g.extend([y[preflen+1:]])
    g = [".".join(x.split(sep)[:-1]) for x in g]
    g = [i for i in g if not i.endswith('.tests')]
#    try:
#        g.remove('ISPy')
#    except:
#        pass
    g = list(set(g))
    g.sort()
    return g

if __name__ == '__main__':
    g = generate_module_list()
    print("modules = [")
    for x in g:
        print("    '%s'," % x)
    print("]")
