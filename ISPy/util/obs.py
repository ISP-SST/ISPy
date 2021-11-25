import numpy as np
from datetime import datetime

from astropy.coordinates import SkyCoord
import astropy.units as u
try:
    from sunpy.coordinates import frames
    import sunpy.coordinates.sun as sun
    sunpylib = True
except:
    sunpylib = False
    print("[ISPy INFO] Install the sunpy package for an accurate implementation.")


def viewangle(xy, date=None, heliographic=False):
    """
    Return the viewing angle theta (and mu = cos(theta)) given a set of solar
    coordinates in the helioprojective or heliographic Stonyhurst system

    Parameters
    ----------
    xy : array_like
        2-element list or array with the coordinate values (in arcsec if
        helioprojective, in degrees if heliographic)
    date : str, optional
        date for which to get the viewing angle (default: today's date)
    heliographic : bool, optional
        switch to indicate input coordinates are heliographic Stonyhurst
        (default False, i.e. coordinates are helioprojective)

    Returns
    -------
    viewangle : list
        2-element list with [theta, mu]. Theta is given in degrees.

    Example
    -------
    >>> from ISPy.util import obs
    >>> result = obs.viewangle([240,-380], date='2017-08-02')
    >>> result = obs.viewangle([30,-50], date='2019-01-12', heliographic=True) # S50 W30

    :Author:
        Gregal Vissers (ISP/SU 2019)
    """

    if date is None:
        date = datetime.now().date().strftime('%Y-%m-%d')

    if sunpylib is True:
        # Convert to helioprojective if input is heliographic Stonyhurst
        if heliographic is True:
            c = SkyCoord(xy[0]*u.deg, xy[1]*u.deg,
                    frame=frames.HeliographicStonyhurst, obstime=date)
            xy_hpc = c.transform_to(frames.Helioprojective)
            xy = [xy_hpc.Tx.value, xy_hpc.Ty.value]

        r_sun = sun.angular_radius(date).value
    else:
        r_sun = 960.469 # Average radius in arcsec
    
    rho = np.sqrt(xy[0]**2 + xy[1]**2)
    if (rho > r_sun):
        raise ValueError("viewangle: coordinates ({0},{1}) are not on the solar "
                + "disc.".format(xy[0],xy[1]))
    mu = np.sqrt(1 - (rho/r_sun)**2)
    theta = np.degrees(np.arccos(mu))

    return [theta, mu]
