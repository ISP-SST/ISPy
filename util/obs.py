import numpy as np
from datetime import datetime

import sunpy.coordinates.sun as sun

def viewangle(xy, date=None):
    """
    Return the viewing angle theta (and mu=cos(theta)) given a set of solar
    (X,Y) coordinates in arcsec

    Arguments:
        xy: 2-element list or array with the (X,Y) coordinate values (in arcsec)

    Keyword arguments:
        date: date for which to get the viewing angle (default: today's date)

    Returns:
        2-element list with [theta, mu]. Theta is given in degrees.

    Example:
        result = viewangle([240,-380], date='2017-08-02')

    Author:
        Gregal Vissers (ISP/SU 2019)
    """

    if date is None:
        date = datetime.now().date().strftime('%Y-%m-%d')

    r_sun = sun.angular_radius(date).value
    rho = np.sqrt(xy[0]**2 + xy[1]**2)
    if (rho > r_sun):
        raise ValueError("viewangle: coordinates ({0},{1}) are not on the solar "
                + "disc.".format(xy[0],xy[1]))
    mu = np.sqrt(1 - (rho/r_sun)**2)
    theta = np.degrees(np.arccos(mu))

    return [theta, mu]
