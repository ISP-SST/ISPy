import numpy as np
import astropy.table 
from scipy.interpolate import interp1d

from ipdb import set_trace as stop
import matplotlib.pyplot as plt

import atlas 

def spectrum(spec, wave, average_profile=None, cgs=True,
        si=False, perHz=True, calib_wave=False, wave_ref=None,
        wave_idx=None):
    """
    Calibrate spectrum intensity in SI or cgs units

    Arguments:
        spec: 1D array with intensity profile in data counts to calibrate. If
            keyword argument `average_profile` is None, then `spec` will be used for
            the intensity (and wavelength) calibration.
        wave: 1D array with wavelengths. Must be of same size as spec.

    Keyword arguments:
        average_profile: averaged intensity profile to use for calibration
            (default None -> use `spec` to calibrate on)
        cgs: output calibration in cgs units (default True)
        si: output calibration in SI units (default False)
        perHz: output calibration per frequency unit (default True)
        calib_wave: perform wavelength calibration prior to intensity
            calibration (default False)
        wave_ref: reference wavelength to clip around in determining line centre
            wavelength for the wavelength calibration (default None -> determine
            from profile)
        wave_idx: wavelength indices to determine the average calibration offset
            over (default None -> use all wavelengths)

    Returns:
        wave: calibrated wavelength array
        spec: calibrated intensity profile array
        factor: offset factor converting data counts to absolute intensity
        spec_fts: atlas profile at wavelengths given by `wave`
        unit: intensity units

    Example:
        wave_cal, spec_cal, factor, spec_fts, units = calib.spectrum(ispec,
            wave, cgs=True, calib_wave=True, wave_idx=[0,1,-2,-1]

    Author:
        Gregal Vissers (ISP/SU 2019)
    """

    wave = np.copy(wave)
    spec = np.copy(spec)
    if average_profile is not None:
        profile = average_profile
    else:
        profile = spec
    
    if wave_idx is None:
        wave_idx = np.arange(wave.size)

    # Get atlas profile for range +/- 0.3
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
    """
    Calibrate spectrum in SI or cgs units

    Arguments:
        spec: 1D array with intensity profile.
        wave: 1D array with wavelengths. Must be of same size as spec.
        spec_fts: 1D array with atlas profile
        wave_fts: 1D array with atlas profile wavelengths. Must be of same size
            as spec_fts.

    Keyword arguments:
        wave_ref: reference wavelength to clip around in determining line centre
            wavelength for the wavelength calibration (default None -> determine
            from profile)
        dwave_ref: clipping range around reference wavelength to determine line
            centre minimum from (default 0.2)

    Returns:
        wave: calibrated wavelength array

    Example:
        wave_cal = wavelength(spec, wave, spec_fts, wave_fts)j

    Author:
        Gregal Vissers (ISP/SU 2019)
    """
  
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
    
