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


def get_paths(level=15):
    """
    Generates a set of paths for modules searching.

    Examples
    ========

    >>> get_paths(2)
    ['ISPy/__init__.py', 'ISPy/*/__init__.py', 'ISPy/*/*/__init__.py']
    >>> get_paths(6)
    ['ISPy/__init__.py', 'ISPy/*/__init__.py', 'ISPy/*/*/__init__.py',
    'ISPy/*/*/*/__init__.py', 'ISPy/*/*/*/*/__init__.py',
    'ISPy/*/*/*/*/*/__init__.py', 'ISPy/*/*/*/*/*/*/__init__.py']

    """
    wildcards = ["/"]
    for i in range(level):
        wildcards.append(wildcards[-1] + "*/")
    p = ["ISPy" + x + "__init__.py" for x in wildcards]
    return p

def generate_module_list(with_cython=False, with_ext=False):
    """
    Generates a list of all available modules

    When with_cython and/or with_ext are provided, the list restricts
    to those modules that require cython and/or the compilations of an
    extension.
    """
    g = []
    for x in get_paths():
        for y in glob(x):
            ok_cython = True
            ok_ext = True
            if len(glob(join(dirname(y),'*.pyx'))) is 0:
                ok_cython = not with_cython
            if not isfile(join(dirname(y),'__extensions__.ispy')):
                ok_ext = not with_ext
            if ok_cython and ok_ext:
                g.extend([y])
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
