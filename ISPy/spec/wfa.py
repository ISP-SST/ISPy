import numpy as np

def cder(x, y):
    """
    Get the finite difference derivative of Stokes I with respect to wavelength

    Parameters
    ----------
    x : array_like
        1D array with the wavelengths
    y : ndarray
        4D nummpy array (ny, nx, nstokes, nwave) with the observations

    Returns
    -------
    cder : ndarray
        3D array with finite difference derivative of Stokes I 

    Example
    --------
    >>> cd = cder(wave, data)

    :Author:
        Jaime de la Cruz Rodríguez (ISP/SU 2018)
    """

    ny, nx, nstokes, nlam = y.shape[:]
    yp = np.zeros((ny, nx, nlam), dtype='float32')

    odx = x[1]-x[0]; ody = (y[:,:,0,1] - y[:,:,0,0]) / odx
    yp[:,:,0] = ody

    for ii in range(1,nlam-1):
        dx = x[ii+1] - x[ii]
        dy = (y[:,:,0,ii+1] - y[:,:,0,ii]) / dx

        yp[:,:,ii] = (odx * dy + dx * ody) / (dx + odx)

        odx = dx; ody = dy

    yp[:,:,-1] = ody
    return yp


class line:
    r"""

    Parameters
    ----------
    cw : integer, optional
       wavelength identifier for pre-defined initialisation 

    Attributes
    ----------
    cw : float
       central wavelength
    j1 : float
        total angular momentum quantum number of the lower level
    j2 : float
        total angular momentum quantum number of the upper level
    g1 : float
       Landé g-factor of the lower level
    g2 : float
       Landé g-factor of the upper level
    geff : float
        effective Landé g-factor
    Gg : float
        \\bar{G} in Eqs. (9.74cd), (9.76) and (9.77) in Degl'Innocenti & Landolfi
        (2004); a measure of the sensitivity of a given spectral line to the
        Zeeman effect in Q and U.
    larm : float, constant
        Larmor frequency

    Example
    -------
    >>> lin = line(6302)
    line::init: cw=6302.4931, geff=2.49, Gg=6.200100000000001

    :Author:
        Jaime de la Cruz Rodríguez (ISP/SU 2018)
    """

    def __init__(self, cw=0):

        self.larm = 4.668645048281451e-13

        if(cw == 8542):
            self.j1 = 2.5; self.j2 = 1.5; self.g1 = 1.2; self.g2 = 1.33; self.cw = 8542.091
        elif(cw == 6301):
            self.j1 = 2.0; self.j2 = 2.0; self.g1 = 1.84; self.g2 = 1.50; self.cw = 6301.4995
        elif(cw == 6302):
            self.j1 = 1.0; self.j2 = 0.0; self.g1 = 2.49; self.g2 = 0.0; self.cw = 6302.4931
        elif(cw == 8468):
            self.j1 = 1.0; self.j2 = 1.0; self.g1 = 2.50; self.g2 = 2.49; self.cw = 8468.4059
        else:
            print("line::init: Warning, line not implemented")
            self.j1 = 0.0; self.j2 = 0.0; self.g1 = 0.0; self.g2 = 0.0; self.cw = 0.0
            return

        j1 = self.j1; j2 = self.j2; g1 = self.g1; g2 = self.g2

        d = j1 * (j1 + 1.0) - j2 * (j2 + 1.0);
        self.geff = 0.5 * (g1 + g2) + 0.25 * (g1 - g2) * d;
        ss = j1 * (j1 + 1.0) + j2 * (j2 + 1.0);
        dd = j1 * (j1 + 1.0) - j2 * (j2 + 1.0);
        gd = g1 - g2;
        self.Gg = (self.geff * self.geff) - (0.0125  * gd * gd * (16.0 * ss - 7.0 * dd * dd - 4.0));

        print("line::init: cw={0}, geff={1}, Gg={2}".format(self.cw, self.geff, self.Gg))

    def update(self, j1, j2, g1, g2, cw):
        """
        Update the atomic data for lines that are not defined in the constructor

        Parameters
        ----------
        j1 : float
            total angular momentum quantum number of the lower level
        j2 : float
            total angular momentum quantum number of the upper level
        g1 : float
           Landé g-factor of the lower level 
        g2 : float
           Landé g-factor of the upper level 
        cw : float
            central wavelength of the line

        Example
        -------
        >>> lin = line(cw=0)
        >>> lin.update(1.0, 0.0, 2.5, 0.0, 6173.3340)
        line::init: cw=6173.334, geff=2.5, Gg=6.25

        :Author:
            Jaime de la Cruz Rodríguez (ISP/SU 2018)
        """

        self.j1 = j1; self.j2 = j2; self.g1 = g1; self.g2 = g2; self.cw = cw

        d = j1 * (j1 + 1.0) - j2 * (j2 + 1.0);
        self.geff = 0.5 * (g1 + g2) + 0.25 * (g1 - g2) * d;
        ss = j1 * (j1 + 1.0) + j2 * (j2 + 1.0);
        dd = j1 * (j1 + 1.0) - j2 * (j2 + 1.0);
        gd = g1 - g2;
        self.Gg = (self.geff * self.geff) - (0.0125  * gd * gd * (16.0 * ss - 7.0 * dd * dd - 4.0));

        print("line::init: cw={0}, geff={1}, Gg={2}".format(self.cw, self.geff, self.Gg))


def getBlos(w, d, line, beta = 0.0):
    """
    Compute the line-of-sight component of the magnetic field vector

    Parameters
    ----------
    w : array_like
        1D numpy array with the wavelength offsets on the observation (can be relative to line-center)
    d : ndarray
        4D numpy array (ny, nx, nstokes, nwav) with the observations
    line : object
        object from wfa.line class with the atomic data of the transition
    beta : float, optional
        regularization factor that selects low-norm solutions (default 0.0). Use wisely.

    Returns
    -------
    Blos : ndarray
        2D array (ny, nx) with the line-of-sight magnetic field

    Example
    -------
    >>> lin = line(8542)
    >>> blos = getBlos(wave, data, lin)

    :Author:
        Jaime de la Cruz Rodriguez (ISP/SU 2018)
    """

    c = -line.larm * line.cw**2 * line.geff; cc = c*c
    der = cder(w, d)

    Blos = (c*d[:,:,3,:] * der).sum(axis=2) / (cc*(der**2).sum(axis=2) + beta)
    return Blos


def getBlosProf(w, d, line, beta = 0.0):

    c = -line.larm * line.cw**2 * line.geff; cc = c*c
    der = cder(w, d)

    Blos = (c*d[:,:,3,:] * der).sum(axis=2) / (cc*(der**2).sum(axis=2) + beta)
    return Blos, c*der*Blos


def getBlosMask(w, d, line, mask, beta = 0.0):
    """
    Compute the line of sight component of the magnetic field vector, using
    selected wavelength points of the observation 

    Parameters
    ----------
    w : array_like
        1D numpy array with the wavelength offsets on the observation (can be relative to line-center)
    d : ndarray
        4D numpy array (ny, nx, nstokes, nwav) with the observations
    line : object
        object from wfa.line class (also included) with the atomic data of the transition
    mask : tuple
        a tuple/numpy array with the wavlength indices that should be used to compute Blong
    beta : float, optional 
        regularization factor that selects low-norm solutions (default 0.0). Use wisely.

    Returns
    -------
    Blos : ndarray
        2D array (ny,nx) with the line-of-sight magnetic field

    Example
    -------
    >>> lin = line(8542)
    >>> mask = (3,4,5,6,7,8,9)
    >>> blos = getBlosMask(wave, data, lin, mask)

    :Author:
        Jaime de la Cruz Rodriguez (ISP/SU 2018)
    """

    c = -line.larm * line.cw**2 * line.geff; cc = c*c
    der = cder(w, d)

    Blos = (c*d[:,:,3,mask] * der[:,:,mask]).sum(axis=2) / (cc*(der[:,:,mask]**2).sum(axis=2) + beta)
    return Blos

def getBhor(w, d, lin, beta = 0.0, vdop = 0.05):
    """
    Compute the horizontal component of the magnetic field vector

    Parameters
    ----------
    w : array_like
        1D numpy array with the wavelength offsets on the observation (can be relative to line-center)
    d : ndarray
        4D numpy array (ny, nx, nstokes, nwav) with the observations
    line : object
        object from wfa.line class with the atomic data of the transition
    beta : float, optional
        regularization factor that selects low-norm solutions. Use wisely.
    vdop : float, optional
        typical Doppler width of the line. Wavelength points inside +/- vdop
        from line center will be excluded (default 0.05)

    Returns
    -------
    Bhor : ndarray
        2D array (ny,nx) with the transverse magnetic field

    Example
    -------
    >>> lin = line(6302)
    >>> blos = getBhor(wave, data, lin, vdop=0.04)

    :Author:
        Jaime de la Cruz Rodriguez (ISP/SU 2018)
    """

    c = 0.75 * (lin.larm * lin.cw**2)**2 * lin.Gg; cc=c*c
    der = cder(w,d)

    for ii in range(len(w)):
        if(np.abs(w[ii]) >= vdop): scl = 1./w[ii]
        else: scl = 0.0
        der[:,:,ii] *= scl


    Q_term = (d[:,:,1,:] * der).sum(axis=2)
    U_term = (d[:,:,2,:] * der).sum(axis=2)

    Bhor = ((c*(Q_term**2 + U_term**2)**0.5) / (cc * (der*der).sum(axis=2) + beta))**0.5
    return Bhor
