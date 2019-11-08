import numpy as np
import astropy.table 
from scipy.interpolate import interp1d

from ipdb import set_trace as stop
import matplotlib.pyplot as plt

import atlas 

def spectrum(spec, wave, average_profile=None, cgs=True,
        si=False, perHz=True, calib_wave=False, wave_ref=None,
        wave_idx=None):

    wave = np.copy(wave)
    spec = np.copy(spec)
    if average_profile is not None:
        profile = average_profile
    else:
        profile = spec
    
    if wave_idx is None:
        wave_idx = np.arange(wave.size)

    # Get atlas profile for range
    fts = atlas.atlas()
    wave_fts, spec_fts, cont_fts = fts.get(wave[0]-0.3, wave[-1]+0.3, cgs=cgs, si=si, perHz=perHz)

    # Calibrate wavelength
    if calib_wave is True:
        wave = wavelength(profile, wave, spec_fts, wave_fts, wave_ref=wave_ref) 

    spec_fts_sel = []
    for ww in range(wave.size):
        widx = np.argmin(np.abs(wave_fts - wave[ww]))
        spec_fts_sel.append(spec_fts[widx])
    offset_factors = 1. / (profile / spec_fts_sel)
    factor = offset_factors[wave_idx].mean()

    spec *= factor

    return wave, spec, factor, spec_fts_sel, fts.sunit



def wavelength(spec, wave, spec_fts, wave_fts, wave_ref=None, dwave_ref=0.2):
  
    wave_spacing = np.diff(wave).mean()

    if wave_ref is None:
        wave_ref = wave[wave.size//2]

    wave_fine = np.arange((wave.size-1)*100.+1)*wave_spacing/100. + wave[0]
    spline = interp1d(wave, spec, kind='cubic')
    spec_fine = spline(wave_fine)
    widx = np.where((wave_fine >= wave_ref-dwave_ref) & \
            (wave_fine <= wave_ref+dwave_ref))[0]
    wave_min = wave_fine[widx[np.argmin(spec_fine[widx])]]
    
    widx = np.where((wave_fts >= wave_ref-dwave_ref) & \
            (wave_fts <= wave_ref+dwave_ref))[0]
    wave_fts_min = wave_fts[widx[np.argmin(spec_fts[widx])]]

    wave += (wave_fts_min - wave_min)

    return wave
    
