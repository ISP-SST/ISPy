"""
Convenience wrappers around glob.glob() for common file types.
"""

import os.path as path
import glob
from ipdb import set_trace as stop

def files(extension, path='./', pattern='', verbose=True):
    """
    Find files with certain pattern and extension in specified path.

    Arguments:
        extension:  extension of the files to list, excluding '.' marker.

    Keyword arguments:
        path: path to directory to search in (default './')
        pattern: pattern in filenames to filter on (default '')
        verbose: verbosity switch controlling printout of search results
            (default True)

    Returns:
        A list with absolute paths to the files found.

    Example:
        f = find.files('fits', path='~/Desktop', pattern='aia304')

    Author:
        Gregal Vissers (ISP/SU 2019)
    """

    lsfiles = glob.glob(path+'*'+pattern+'*'+extension)

    if verbose is True:
        if len(lsfiles) > 0:
            print("Searched "+path)
            for nn in range(len(lsfiles)):
                print("{0} {1}".format(nn,path.basename(lsfiles[nn])))
        else:
            print("files: No files found with pattern {0}*{1} in {2}".format(pattern, extension, path))
    
    return lsfiles

def nc(path='./', pattern='', verbose=True):
    lsfiles = files('nc', path=path, pattern=pattern, verbose=verbose)

    return lsfiles

def fits(path='./', pattern='', verbose=True):
    lsfiles = files('fits', path=path, pattern=pattern, verbose=verbose)

    return lsfiles

def cube(path='./', pattern='', verbose=True):
    lsfiles = files('cube', path=path, pattern=pattern, verbose=verbose)

    return lsfiles

def idlsave(path='./', pattern='', verbose=True):
    lsfiles = files('idlsave', path=path, pattern=pattern, verbose=verbose)

    return lsfiles
