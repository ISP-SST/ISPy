import numpy as np
import os
import scipy.io.idl as idl
import astropy.units as u
import astropy.constants as const

class atlas:
    """
    Class to load (FTS) spectral atlas

    Parameters
    ----------
    None

    Attributes
    ----------
    cont : array_like
        full atlas continuum intensities in units `sunit`
    spec : array_like
        full atlas spectrum in units `sunit`
    wave : array_like
        wavelengths in units `wunit`
    usys : str
        unit system, one of `si_inu`, `si_ilambda`, `cgs_inu` or `cgs_ilambda`
    sunit : astropy CompositeUnit object
        intensity units according to the `usys` setting
    wunit : astropy Unit object
        wavelength unit

    Example
    -------
    >>> from ISPy.spec import atlas as S
    >>> fts = S.atlas()
    >>> wav, sp, cont = fts.get(6562.,6564., cgs=True, perHz=False)

    :Author:
        Jaime de la Cruz Rodriguez (ISP/SU 2019)
    """
    def __init__(self):
        # Check dir where this class is stored
        this_dir, this_filename = os.path.split(__file__)
        DATA_PATH = os.path.join(this_dir, "../data/fts_disk_center_SI.idlsave")

        # Load data file
        fts = idl.readsav(DATA_PATH)
        self.cont = np.copy(fts["ftscnt_SI"])
        self.spec = np.copy(fts["ftsint_SI"])
        self.wave = np.copy(fts["ftswav"])
        self.usys = 'si_inu' # J/s/m^2/sr/Hz
        self.sunit = u.J / (u.s * u.m**2 * u.steradian * u.Hz)
        self.wunit = u.Angstrom


    def to(self, usys_to, perHz=True):
        """
        Convert atlas intensity data to particular units

        Parameters
        ----------
        usys_to : str
            descriptor setting the unit system to convert to, either `si` or
            `cgs` (case insensitive)
        perHz : bool, optional
            convert to intensity units per Hz (defaults True)

        Example
        -------
        >>> from ISPy.spec import atlas as S
        >>> fts = S.atlas() # intensity units are J/s/m^2/sr/Hz (SI, per Hz) by default
        >>> fts.to('cgs', perHz=False) # convert to erg/s/cm^2/sr/A

        :Author:
            Gregal Vissers (ISP/SU 2020)

        """
        usys_from = self.usys.lower()

        # Determine SI <-> cgs conversion
        if usys_to.lower() == 'cgs' and usys_from[:2] == 'si':
            conversion = u.J.to('erg') / (u.m.to('cm')**2)
            self.sunit *= u.m**2 / u.J * u.erg / u.cm**2
        elif usys_to.lower() == 'si' and usys_from[:3] == 'cgs':
            conversion = u.erg.to('J') / (u.cm.to('m')**2)
            self.sunit *= u.cm**2 / u.erg * u.J / u.m**2
        else:
            conversion = 1.

        # Apply I_lambda (per AA) <-> I_nu (per Hz) if need be
        lambda_to_nu = (self.wave*u.Angstrom.to('m'))**2 / const.c.value
        if (perHz == False and usys_from[-3:] != 'inu') or \
            (perHz == True and usys_from[-3:] == 'inu'):
            # no change to conversion factor
            if perHz == True:
                ext = '_inu'
            else:
                ext = '_ilambda'
        elif (perHz == False and usys_from[-3:] == 'inu'):
            conversion /= lambda_to_nu
            self.sunit *= u.Hz / u.Angstrom
            ext = '_ilambda'
        else:
            conversion *= lambda_to_nu
            self.sunit *= u.Angstrom / u.Hz
            ext = '_inu'

        # Apply conversion and update current unit system
        self.spec *= conversion
        self.cont *= conversion
        self.usys = usys_to + ext

    def get(self, w0, w1, cgs=False, nograv=False, perHz=True):
        """
        Extract a subset of the atlas profile

        Parameters
        ----------
        w0, w1: float
            lower and upper boundary of the wavelength range for which to
            extract the atlas profile
        cgs : bool, optional
            return the intensities in cgs units (defaults False, i.e. use SI)
        nograv : bool, optional
            account for gravitationl reddening (defaults False)
        perHz : bool, optional
            return intensity in units per Hz (defaults True)
        
        Example
        -------
        See class docstring

        :Authors:
            Jaime de la Cruz Rodriguez (ISP/SU 2019), Gregal Vissers (ISP/SU
            2020)
        """

        idx = (np.where((self.wave >= w0) & (self.wave <= w1)))[0]
        
        if cgs is True:
            self.to('cgs', perHz=perHz)
        else:
            self.to('si', perHz=perHz)

        wave = np.copy(self.wave[idx[0]:idx[-1]])
        spec = np.copy(self.spec[idx[0]:idx[-1]])
        cont = np.copy(self.cont[idx[0]:idx[-1]])

        if(not nograv):
            wave *=  (1.0-633.0/const.c.value) # grav reddening

        # Normalize by the continuum if cgs=False and si=False (default)
        if (not cgs and not si):
            spec /= cont
            cont[:] = 1.0
            
        return wave, spec, cont
