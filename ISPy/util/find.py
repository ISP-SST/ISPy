"""
Convenience wrappers around glob.glob() to list common file types.
"""

import os.path 
import glob

def files(extension, path='./', pattern='', verbose=True):
    """
    Find files with certain pattern and extension in specified path.

    Parameters
    ----------
    extension : str
        extension of the files to list, excluding '.' marker.
    path : str, optional
        path to directory to search in (default './')
    pattern: str, optional
        pattern in filenames to filter on (default '')
    verbose: bool, optional
        verbosity switch controlling printout of search results (default True)

    Returns
    -------
    lsfiles : list
        A list with absolute paths to the files found.

    Example
    -------
    >>> f = find.files('fits', path='~/Desktop', pattern='aia304')

    :Author:
        Gregal Vissers (ISP/SU 2019)
    """

    if extension:
        if len(path) == 0: path = './'
        if path[-1] != '/': path += '/'

        lsfiles = glob.glob(path+'*'+pattern+'*'+extension)
        # glob.glob() will give you a list with an arbitrary order
        lsfiles = sorted(lsfiles)

        if verbose is True:
            if len(lsfiles) > 0:
                print("Searched "+path)
                for nn in range(len(lsfiles)):
                    print("{0} {1}".format(nn,os.path.basename(lsfiles[nn])))
            else:
                print("files: No files found with pattern {0}*{1} in {2}".format(pattern, extension, path))
    else:
        raise ValueError("files: extension cannot be an empty string.")
        
    return lsfiles

def nc(path='./', pattern='', verbose=True):
    """ Convenience wrapper around find.files to get *.nc files """
    return files('nc', path=path, pattern=pattern, verbose=verbose)

def fits(path='./', pattern='', verbose=True):
    """ Convenience wrapper around find.files to get *.fits files """
    return files('fits', path=path, pattern=pattern, verbose=verbose)

def cube(path='./', pattern='', verbose=True):
    """ Convenience wrapper around find.files to get *.cube files """
    return files('cube', path=path, pattern=pattern, verbose=verbose)

def idlsave(path='./', pattern='', verbose=True):
    """ Convenience wrapper around find.files to get *.idlsave files """
    return files('idlsave', path=path, pattern=pattern, verbose=verbose)
