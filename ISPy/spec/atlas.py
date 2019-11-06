import numpy as np
import os
import scipy.io.idl as idl
import astropy.units as u
import astropy.constants as const

class atlas:
    """
    Class to load (FTS) spectral atlas

    Methods:
        __init__()
        tocgs(w, s)
        tosi(w, s)
        get(w0, w1, cgs=False, si=False, nograv=False)

    Example:
        import atlas as S
        fts = S.atlas()
        wav, sp, cont = fts.get(6562.,6564., cgs=True)

    Author:
        Jaime de la Cruz Rodriguez (ISP/SU 2019)
    """
    def __init__(self):
        # Check dir where this class is stored
        this_dir, this_filename = os.path.split(__file__)
        DATA_PATH = os.path.join(this_dir, "../data/fts_disk_center.idlsave")

        # Load data file
        fts = idl.readsav(DATA_PATH)
        self.cont = fts["ftscnt"]
        self.spec = fts["ftsint"]
        self.wave = fts["ftswav"]


    def tocgs(self, w, s):
        clight = const.c.to('cm/s').value #speed of light [cm/s]
        joule_2_erg = u.J.to('erg')
        aa_to_cm = u.Angstrom.to('cm')
        s *= joule_2_erg/aa_to_cm # from Watt /(cm2 ster AA) to erg/(s cm2 ster cm)
        s *= (w*aa_to_cm)**2/clight   # to erg/
        return s

    def tosi(self, wav, s):
        clight = const.c.value #speed of light [m/s]                                  
        aa_to_m = u.Angstrom.to('m')
        cm_to_m = u.cm.to('m')
        s /= cm_to_m**2 * aa_to_m # from from Watt /(s cm2 ster AA) to Watt/(s m2 ster m) 
        s *= (wav*aa_to_m)**2 / clight # to Watt/(s m2 Hz ster)
        return s
    
    def get(self, w0, w1, cgs = False, si = False, nograv = False):
        idx = (np.where((self.wav >= w0) & (self.wav <= w1)))[0]

        wave = np.copy(self.wave[idx[0]:idx[-1]])
        spec = np.copy(self.spec[idx[0]:idx[-1]])
        cont = np.copy(self.cont[idx[0]:idx[-1]])

        if(not nograv):
            wave *=  (1.0-633.0/const.c.value) # grav reddening

        # convert to CGS units
        if(cgs):
            spec = self.tocgs(wave, spec)
            cont = self.tocgs(wave, cont)

        # convert to IS units
        elif(si):
            spec = self.tosi(wave, spec)
            cont = self.tosi(wave, cont)

        # Normalize by the continuum (default)
        else:
            spec /= cont
            cont[:] = 1.0
            
        return wave, spec, cont
