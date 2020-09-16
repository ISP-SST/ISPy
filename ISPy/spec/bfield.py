import numpy as np
import astropy.units as u

DTOR = u.deg.to('rad')

def observer2spherical(bvec_in, degrees=False):
    """
    Convert the magnetic field vector in the observer's frame B(lon, hor, azi)
    to spherical coordinate system B(field, inc, azi)

    Parameters
    ----------
    bvec_in : ndarray
        magnetic field vector of shape (3,), (3, nx) or (3, ny, nx)
    degrees : bool, optional
        return the inclination and azimuth in degrees (default False = radians)

    Returns
    -------
    ndarray
        magnetic field vector of same shape as `bvec_in`, converted to spherical
        coordinates

    Examples
    --------
    >>> import bfield
    >>> bvec.shape
    (3, 374, 175)
    >>> bvec[2].min(), bvec[2].max()
    (0.000711255066562444, 179.99998474121094)
    >>> bsph = bfield.observer2spherical(bvec, degrees=True)
    >>> bsph[2].min(), bsph[2].max()
    (1.2413742733006074e-05, 3.1415923872736844)

    """

    bvec_out = np.copy(bvec_in)
    bvec_out[0] = np.sqrt(bvec_in[0]**2 + bvec_in[1]**2)
    bvec_out[1] = np.arccos(bvec_in[0]/bvec_out[0])
    if bvec_in[2].max() > 2*np.pi: bvec_out[2] *= DTOR

    if degrees is True:
        bvec_out[1] /= DTOR
        if bvec_in[2].max() <= 2*np.pi: bvec_out[2] /= DTOR

    return bvec_out

def spherical2observer(bvec_in, degrees=False):
    """
    Convert the magnetic field vector in spherical coordinate system B(field,
    inc, azi) to the observer's frame B(lon, hor, azi)
    
    Parameters
    ----------
    bvec_in : ndarray
        magnetic field vector of shape (3,), (3, nx) or (3, ny, nx)
    degrees : bool, optional
        return the azimuth in degrees (default False = radians)

    Returns
    -------
    ndarray
        magnetic field vector of same shape as `bvec_in`, converted to the
        observer's frame

    Examples
    --------
    >>> bobs = bfield.spherical2observer(bvec)
    
    """
    
    bvec_out = np.copy(bvec_in)
    if bvec_in[1].max() > np.pi: bvec_in[1] *= DTOR
    bvec_out[0] = bvec_in[0] * np.cos(bvec_in[1])
    bvec_out[1] = np.sqrt(bvec_in[0]**2 - bvec_out[0]**2)
    
    if (degrees is True) and (bvec_in[2].max() <= 2*np.pi): 
        bvec_out[2] /= DTOR

    return bvec_out

def spherical2cartesian(bvec_in, azim0=0):
    """
    Convert the magnetic field vector in spherical coordinate system B(field,
    inc, azi) to Cartesian coordinates B(x, y, z)

    Parameters
    ----------
    bvec_in : ndarray
        magnetic feld vector of shape (3,), (3, nx) or (3, ny, nx)
    azim0 : {0, 1, 2, 3}, optional
        zero-azimith direction convention: +Y (azim0=0), +X (azim0=1), -Y
        (azim0=2), -X (azim0=3)

    Returns
    -------
    ndarray
        magnetic field vector of same shape as `bvec_in`, converted to Cartesian
        coordinates
    
    Examples
    --------
    >>> bxyz = bfield.spherical2cartesian(bvec)

    """

    bvec_out = np.copy(bvec_in)
    if bvec_in[1].max() > np.pi: bvec_in[1] *= DTOR
    if bvec_in[2].max() > 2*np.pi: bvec_in[2] *= DTOR 

    #Zero-azimuth direction
    if azim0 == 0:
        azim_x = -np.sin(bvec_in[2])
        azim_y = np.cos(bvec_in[2])
    elif azim0 == 1:
        azim_x = np.cos(bvec_in[2])
        azim_y = np.sin(bvec_in[2])
    elif azim0 == 2:
        azim_x = np.sin(bvec_in[2])
        azim_y = -np.cos(bvec_in[2])
    elif azim0 == 3:
        azim_x = -np.cos(bvec_in[2])
        azim_y = -np.sin(bvec_in[2])
    else:
        raise ValueError('spherical2cartesian: azim0 value invalid')

    bvec_out[0] = bvec_in[0] * np.sin(bvec_in[1]) * azim_x
    bvec_out[1] = bvec_in[0] * np.sin(bvec_in[1]) * azim_y
    bvec_out[2] = bvec_in[0] * np.cos(bvec_in[1])
    
    return bvec_out

def cartesian2spherical(bvec_in, azim0=0, degrees=False):
    """
    Convert the magnetic field vector in Cartesian coordinates B(x, y, z) to
    spherical coordinates B(field, inc, azi)

    Parameters
    ----------
    bvec_in : ndarray
        magnetic feld vector of shape (3,), (3, nx) or (3, ny, nx)
    azim0 : {0, 1, 2, 3}, optional
        zero-azimith direction convention: +Y (azim0=0), +X (azim0=1), -Y
        (azim0=2), -X (azim0=3)
    degrees : bool, optional
        return the inclination and azimuth in degrees (default False = radians)

    Returns
    -------
    ndarray
        magnetic field vector of same shape as `bvec_in`, converted to spherical
        coordinates
    
    Examples
    --------
    >>> bsph = bfield.cartesian2spherical(bvec, degrees=True)

    """

    bvec_out = np.copy(bvec_in)
    bvec_out[0] = np.sqrt(bvec_in[0]**2 + bvec_in[1]**2 + bvec_in[2]**2)
    bvec_out[1] = np.arccos(bvec_in[2] / bvec_out[0])

    # Take zero-azimuth direction into account
    if (azim0 == 0) or (azim0 == 2):
        bvec_out[2] = np.arctan(-1.*bvec_in[0] / (bvec_in[1]+1e-20))  # +1e-20 to avoid ZeroDivisionError
    elif (azim0 == 1) or (azim0 == 3):
        bvec_out[2] = np.arctan(bvec_in[1] / (bvec_in[0]+1e-20))  # +1e-20 to avoid ZeroDivisionError
    else:
        raise ValueError('spherical2cartesian: azim0 value invalid')
    
    # Correct for periodicity in solutions, i.e. map azimuth to [0,2pi)
    # Assumes +Y is zero-azimuth direction, increasing counter-clockwise
    idx = np.where(bvec_in[1] < 0)
    bvec_out[2][idx] += np.pi
    idx = np.where((bvec_in[0] > 0) & (bvec_in[1] > 0))
    bvec_out[2][idx] += 2*np.pi
    
    if degrees is True:
        bvec_out[1] /= DTOR
        bvec_out[2] /= DTOR

    return bvec_out

def observer2cartesian(bvec_in):
    """
    Convert the magnetic field vector in the observer's frame B(lon, hor, azi)
    to Cartesian coordinates B(x, y, z).
    Wrapper function around observer2spherical() and spherical2cartesian().
    
    Parameters
    ----------
    bvec_in : ndarray
        magnetic feld vector of shape (3,), (3, nx) or (3, ny, nx)

    Returns
    -------
    ndarray
        magnetic field vector of same shape as `bvec_in`, converted to Cartesian
        coordinates
    
    Examples
    --------
    >>> bxyz = bfield.observer2cartesian(bvec)

    """
    
    return spherical2cartesian(observer2spherical(bvec_in))

def cartesian2observer(bvec_in, degrees=False):
    """
    Convert the magnetic field vector in Cartesian coordinates B(x, y, z) to the
    observer's frame B(lon, hor, azi)
    Wrapper function around cartesian2spherical() and spherical2observer().
    
    Parameters
    ----------
    bvec_in : ndarray
        magnetic feld vector of shape (3,), (3, nx) or (3, ny, nx)
    degrees : bool, optional
        return the azimuth in degrees (default False = radians)

    Returns
    -------
    ndarray
        magnetic field vector of same shape as `bvec_in`, converted to the
        observer's frame
    
    Examples
    --------
    >>> bobs = bfield.cartesian2observer(bvec, degrees=True)

    """

    return spherical2observer(cartesian2spherical(bvec_in), degrees=degrees)

