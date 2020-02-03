import os

import numpy as np
import astropy.table 
from scipy.interpolate import interp1d
from scipy.ndimage import convolve
from scipy.optimize import differential_evolution
from astropy.io import fits
from astropy.table import Table

import matplotlib.pyplot as plt
from ipdb import set_trace as stop

import atlas 

def get_calibration(wave_obs, spec_obs, wave_atlas, spec_atlas, bounds=None, weights=None):
    """
    Get calibration offsets from fitting `spec_obs` to `spec_atlas`, assuming
    wavelength grids `wave_obs` and `wave_atlas`

    Arguments:
        wave_obs: 1D array with observed wavelengths. Must be of same size as
            `spec_obs`.
        spec_obs: 1D array with observed intensities.
        wave_atlas: 1D array with wavelengths corresponding to `spec_atlas`.
        spec_atlas: 1D array with atlas intensity profile (e.g. from
            `ISpy.spec.atlas`)

    Keyword arguments:
        bounds: list of tuples [(ifact_low, ifact_upp), (woff_low, woff_upp)]
            suggesting lower and upper bounds for fitting the intensity factor
            and wavelength offset (defaults to 1/50th and 50 times the fraction
            of `spec_atlas` to `spec_obs` for ifact, and ±0.3 for woff)
        weights: array of weights of same size as `wave_obs` used in determining
            chi-square (defaults to 1 for every wavelenght position)

    Returns:
        calibration: 2-element array [ifact, woff] with multiplication factor
        and wavelength offset to be applied to `spec_obs` and `wave_obs`
        respectively

    Author: Gregal Vissers, Carlos Diaz Baso (ISP/SU 2020)
    """

    def func_to_optimise(x):
      x0 = x[0]
      x1 = x[1]
      ospec = spec_obs * x0
      atlas = np.interp(wave_obs, wave_atlas-x1, spec_atlas)
      chi2 = np.sum( (atlas-ospec)**2 * weights)
      return chi2

    if bounds is None:
        bounds = [(spec_atlas[0]/spec_obs[0]*0.02, spec_atlas[0]/spec_obs[0]*50.), (-0.3, 0.3)]
    optim = differential_evolution(func_to_optimise, bounds)
    calibration = optim.x

    return calibration


def spectrum(wave, spec, mu=1.0, spec_avg=None, cgs=True, si=False, perHz=True,
    calib_wave=False, wave_idx=None, extra_weight=20., bounds=None,
    instrument_profile=None, verbose=False):
    """
    Calibrate spectrum intensity (in SI or cgs units) and wavelength by
    simultaneously fitting offsets given an atlas profile

    Arguments:
        wave: 1D array with wavelengths. Must be of same size as `spec`.
        spec: 1D array with intensity profile in data counts to calibrate. If
            keyword argument `spec_avg` is None, then `spec` will be used for
            the intensity (and wavelength) calibration.

    Keyword arguments:
        mu: cosine of heliocentric viewing angle of the observations (defaults
            1.0 -> disc centre)
        spec_avg: averaged intensity profile to use for calibration
            (default None -> use `spec` to calibrate on)
        cgs: output calibration in cgs units (default True)
        si: output calibration in SI units (default False)
        perHz: output calibration per frequency unit (default True)
        calib_wave: perform wavelength calibration prior to intensity
            calibration (default False)
        wave_idx: wavelength indices that will get `extra_weight` during while
            fitting the intensity profile (default None -> all wavelengths get
            equal weight)
        extra_weight: amount of extra weight to give selected wavelength
            positions as specified by `wave_idx` (default 20)
        bounds: list of tuples [(ifact_low, ifact_upp), (woff_low, woff_upp)]
            suggesting lower and upper bounds for fitting the intensity factor
            and wavelength offset (defaults None)
        instrument_profile: 2D array with wavelength spacing (starting at 0) and
            instrumental profile to convolve the atlas profile with

    Returns:
        wave: calibrated wavelength array
        spec: calibrated intensity profile array
        factor: offset factor converting data counts to absolute intensity
        spec_fts: atlas profile at wavelengths given by `wave`. Convolved with
            instrument profile if instrument_profile is not None.
        unit: intensity units

    Example:
        wave_cal, spec_cal, factor, spec_fts, units = calib.spectrum(ispec,
            wave, cgs=True, calib_wave=True, wave_idx=[0,1,-2,-1])

    Author:
        Gregal Vissers, Carlos Diaz Baso (ISP/SU 2019-2020)
    """

    wave = np.copy(wave)
    spec = np.copy(spec)
    if spec_avg is not None:
        profile = spec_avg
    else:
        profile = spec
    
    if wave_idx is None:
        wave_idx = np.arange(wave.size)
    else:
        wave_idx = np.atleast_1d(wave_idx)

    # Get atlas profile for range +/- 0.3
    fts = atlas.atlas()
    wave_fts, spec_fts_orig, cont_fts = fts.get(wave[0]-0.3, wave[-1]+0.3, cgs=cgs, si=si, perHz=perHz)

    # Correct for limb-darkening
    limbdark_factor = limbdarkening(wave_fts, mu=mu)
    spec_fts_orig *= limbdark_factor
    cont_fts *= limbdark_factor

    # Apply instrument profile if provided
    if instrument_profile is not None:
        wave_ipr_spacing = np.diff(instrument_profile[:,0]).mean()
        wave_fts_spacing = np.diff(wave_fts).mean()
        nw_ipr = instrument_profile.shape[0]
        wave_ipr_fine = np.arange((nw_ipr-1) * wave_ipr_spacing / wave_fts_spacing + 1) \
                * wave_fts_spacing
        kernel = np.interp(wave_ipr_fine, instrument_profile[:,0],
                instrument_profile[:,1])
        kernel /= np.sum(kernel)
        spec_fts = convolve(spec_fts_orig, kernel, mode='nearest')
    else:
        spec_fts = spec_fts_orig

    weights = np.ones_like(wave)
    if wave_idx.size is not wave.size:
        weights[wave_idx] = extra_weight

    calibration = get_calibration(wave, profile, wave_fts, spec_fts, bounds=bounds, weights=weights)

    # Apply calibration and prepare output
    if calib_wave is True:
        wave += calibration[1]
    spec *= calibration[0]

    spec_fts_sel = []
    for ww in range(wave.size):
        widx = np.argmin(np.abs(wave_fts - wave[ww]))
        spec_fts_sel.append(spec_fts[widx])

    if verbose is True:
        plot_scale_factor = 1.e-5
        fig, ax = plt.subplots()
        legend_items = ('observed profile', 'selected points', 'atlas profile')
        ax.plot(wave, spec/plot_scale_factor, '.')
        ax.plot(wave[wave_idx], spec[wave_idx]/plot_scale_factor, '+')
        ax.plot(wave_fts, spec_fts_orig/plot_scale_factor)
        if instrument_profile is not None:
            ax.plot(wave_fts, spec_fts/plot_scale_factor,'--')
            legend_items += ('atlas convolved with instrument profile',)
        ax.set_ylabel('intensity ['+r'$\times10^{-5}$'+' {0}]'.format(fts.sunit.to_string()))
        ax.set_xlabel('wavelength [{0}]'.format(fts.wunit.to_string()))
        ax.legend(legend_items)
        ax.set_title('ISPy: calib.spectrum() results')
        plt.show()
        if calib_wave is True:
            print("spectrum: wavelength calibration offset: {0} (added to input wavelengths)".format(calibration[1]))
        print("spectrum: intensity calibration offset factor: {0}".format(calibration[0]))

    return wave, spec, calibration, spec_fts_sel, fts.sunit


def limbdarkening(wave, mu=1.0, nm=False):
    """
    Return limb-darkening factor given wavelength and viewing angle
    mu=cos(theta)

    Arguments:
        wave: scalar or 1D array with wavelength(s).

    Keyword arguments:
        mu: cosine of heliocentric viewing angle (default 1.0 -> disc centre)
        nm: input wavelength units are nanometers (default False)

    Returns:
        factor: scaling factor(s) to be applied for given input wavelengths. Has
        as many elements as `wave`.

    Example:
        factor = limbdarkening(630.25, mu=0.7, nm=True)

    Author:
        Gregal Vissers (ISP/SU 2020)
    """

    this_dir, this_filename = os.path.split(__file__)
    DATA_PATH = os.path.join(this_dir, "../data/limbdarkening_Neckel_Labs_1994.fits")

    wave = np.atleast_1d(wave)  # Ensure input is iterable

    table = Table(fits.getdata(DATA_PATH))
    wavetable = np.array(table['wavelength'])
    if nm is False:
        wavetable *= 10.

    # Get table into 2D numpy array
    Atable = np.array([ table['A0'], table['A1'], table['A2'],
        table['A3'], table['A4'], table['A5'] ])
    
    factor = np.zeros((wave.size), dtype='float64')
    for ii in range(6):
      Aint = np.interp(wave, wavetable, Atable[ii,:])
      factor += Aint * mu**ii

    return factor
