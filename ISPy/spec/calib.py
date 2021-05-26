 # -*- coding: utf-8 -*-
import os
import numpy as np
import astropy.table 
from scipy.interpolate import interp1d
from scipy.ndimage import convolve
from scipy.optimize import differential_evolution
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
import matplotlib.pyplot as plt
from ISPy.spec import atlas


def get_calibration(wave_obs, spec_obs, wave_atlas, spec_atlas, mu=1.0,
        calib_at_dc=False, wave_idx=None, extra_weight=20., bounds=None):
    """
    Get calibration offsets from fitting `spec_obs` to `spec_atlas`, assuming
    wavelength grids `wave_obs` and `wave_atlas`

    Parameters
    ----------
    wave_obs : array_like
        1D array with observed wavelengths. Must be of same size as `spec_obs`.
    spec_obs : array_like
        1D array with observed intensities.
    wave_atlas : array_like
        1D array with wavelengths corresponding to `spec_atlas`.
    spec_atlas : array_like
        1D array with atlas intensity profile (e.g. from `ISpy.spec.atlas`)
    mu : float, optional
        cosine of heliocentric viewing angle of the observations (defaults 1.0
        -> disc centre)
    calib_at_dc : bool, optional
        calibrate assuming `spec_avg` (or `spec`, if `spec_avg` is
        None) was taken at disc centre (defaults False).
    wave_idx : array_like, optional
        wavelength indices that will get `extra_weight` during while fitting the
        intensity profile (default None -> all wavelengths get equal weight)
    extra_weight : float, optional
        amount of extra weight to give selected wavelength positions as
        specified by `wave_idx` (default 20)
    bounds : list of tuples, optional
        list of tuples [(ifact_low, ifact_upp), (woff_low, woff_upp)] suggesting
        lower and upper bounds for fitting the intensity factor and wavelength
        offset (defaults to 1/50th and 50 times the fraction of `spec_atlas` to
        `spec_obs` for ifact, and ±0.3 for woff)

    Returns
    -------
    calibration : 2-element array
        multiplication factor and wavelength offset [ifact, woff] to be applied
        to `spec_obs` and `wave_obs` respectively.
    
    Example
    -------
    >>> calibration = get_calibration(wave, spec, wave_atlas, spec_atlas, mu=0.4, calib_at_dc=True)

    :Authors:
        Carlos Diaz (ISP/SU 2020), Gregal Vissers (ISP/SU 2020)

    """

    if wave_idx is None:
        wave_idx = np.arange(wave_obs.size)
    else:
        wave_idx = np.atleast_1d(wave_idx)

    # Correct for limb-darkening if profile to calibrate on is not from
    # disc centre (and presumably at same mu as observations)
    if calib_at_dc is False:
        spec_atlas = spec_atlas * limbdarkening(wave_atlas, mu=mu)

    weights = np.ones_like(wave_obs)
    if wave_idx.size is not wave_obs.size:
        weights[wave_idx] = extra_weight

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

    # Apply limb-darkening correction if calibration was on profile from disc
    # centre and data was not; for calib_at_dc is False, limb-darkening is already in
    # optimised offset
    if (calib_at_dc is True) and (mu != 1.0):
        calibration[0] *= np.mean(limbdarkening(wave_atlas, mu=mu))

    return calibration


def convolve_atlas(wave_atlas, spec_atlas, instrument_profile, mode='nearest'):
    """
    Convolve spectral atlas profile with instrument profile (after interpolation
    to the atlas wavelength grid)

    Parameters
    ----------
    wave_atlas : array_like
        1D array with wavelengths corresponding to `spec_atlas`.
    spec_atlas : array_like
        1D array with atlas intensity profile (e.g. from `ISpy.spec.atlas`)
    instrument_profile : ndarray
        2D array [wave, profile] with wavelength spacing (starting at 0) and
        instrumental profile to convolve the atlas profile with
    mode : str, optional
        set interpolation in call to `np.interp` (defaults 'nearest')

    Returns
    -------
    spec_convolved : array_like
        1D array with convolved profile

    Example
    -------
    >>> convolved = calib.convolve_atlas(wave_atlas, spec_atlas, ipr, mode='cubic')

    :Author: 
        Gregal Vissers (ISP/SU 2020)

    """

    wave_ipr_spacing = np.diff(instrument_profile[:,0]).mean()
    wave_atlas_spacing = np.diff(wave_atlas).mean()
    nw_ipr = instrument_profile.shape[0]
    wave_ipr_fine = np.arange((nw_ipr-1) * wave_ipr_spacing / wave_atlas_spacing + 1) \
            * wave_atlas_spacing
    kernel = np.interp(wave_ipr_fine, instrument_profile[:,0],
            instrument_profile[:,1])
    kernel /= np.sum(kernel)
    spec_convolved = convolve(spec_atlas, kernel, mode=mode)

    return spec_convolved


def spectrum(wave, spec, mu=1.0, spec_avg=None, calib_at_dc=False,
        atlas_range=0.5, wave_idx=None, extra_weight=20., bounds=None,
        instrument_profile=None, calib_wave=False, cgs=True, si=False,
        perHz=True, qsdc_calib=False, verbose=False):
    """
    Calibrate spectrum intensity (in SI or cgs units) and wavelength by
    simultaneously fitting offsets given an atlas profile

    Parameters
    ----------
    wave : array_like
        1D array with wavelengths.
    spec : ndarray
        data (cube) with intensity profile(s) in counts to apply calibration to.
        May be higher dimension cube (e.g. [nt, ny, nx, nwave, nstokes]). If
        keyword argument `spec_avg` is None, then `spec` is assumed to be a 1D
        array of Stokes I intesities of same size as `wave` and will be used to
        determine the intensity calibration offset factor.
    mu : float, optional
        cosine of heliocentric viewing angle of the observations (defaults 1.0
        -> disc centre)
    spec_avg : array_like, optional
        averaged intensity profile to use for calibration
        (default None -> use `spec` to calibrate on)
    calib_at_dc : bool, optional
        calibrate assuming `spec_avg` (or `spec`, if `spec_avg` is None) was
        taken at disc centre (defaults False).
    atlas_range : float, optional
        get atlas profile with for the range +/- this value (defaults 0.5)
    wave_idx: array_like, optional
        wavelength indices that will get `extra_weight` during while fitting the
        intensity profile (default None -> all wavelengths get equal weight)
    extra_weight : float, optional
        amount of extra weight to give selected wavelength positions as
        specified by `wave_idx` (default 20)
    bounds : list of tuples, optional
        [(ifact_low, ifact_upp), (woff_low, woff_upp)] suggesting lower and
        upper bounds for fitting the intensity factor and wavelength offset
        (defaults None)
    instrument_profile : ndarray, optional
        2D array [wave, profile] with wavelength spacing (starting at 0) and
        instrumental profile to convolve the atlas profile with
    calib_wave : bool, optional
        perform wavelength calibration prior to intensity calibration (default
        False)
    cgs : bool, optional
        output calibration in cgs units (default True)
    si : bool, optional
        output calibration in SI units (default False)
    perHz : bool, optional
        output calibration per frequency unit (default True)
    qsdc_calib : bool, optional
        output calibration as fraction of quiet Sun disc centre continuum
        intensity (default False). If set, overrides `cgs`, `si` and `perHz`
    verbose : bool, optional
        output calibration plot and offset values to command line (defaults
        False)

    Returns
    -------
    wave : array_like
        calibrated wavelength array
    spec : array_like
        calibrated intensity profile array
    factor : float
        offset factor converting data counts to absolute intensity
    spec_fts : array_like
        atlas profile at wavelengths given by `wave`. Convolved with
        instrument profile if instrument_profile is not None.
    unit : object
        intensity units

    Example
    -------
    >>> wave_cal, spec_cal, factor, spec_fts, units = calib.spectrum(wave, ispec,
            cgs=True, calib_wave=True, wave_idx=[0,1,-2,-1])

    :Author:
        Gregal Vissers, Carlos Diaz (ISP/SU 2019-2020)

    """

    if spec_avg is not None:
        profile = np.copy(spec_avg)
    else:
        if spec.ndim == 1:
            profile = np.copy(spec)
        else:
            raise ValueError("`spec` must be a 1D array when `spec_avg` is not set")

    # Get atlas profile for range +/- 0.3
    fts = atlas.atlas()
    atlas_range = np.abs(atlas_range)
    wave_fts, spec_fts_dc, cont_fts = fts.get(wave[0]-atlas_range,
            wave[-1]+atlas_range, cgs=cgs, perHz=perHz)

    # Apply instrument profile if provided
    if instrument_profile is not None:
        spec_fts = convolve_atlas(wave_fts, spec_fts_dc, instrument_profile)
    else:
        spec_fts = np.copy(spec_fts_dc)

    # Get calibration offset factor and shift
    calibration = get_calibration(wave, profile, wave_fts, spec_fts,
            bounds=bounds, calib_at_dc=calib_at_dc, mu=mu, wave_idx=wave_idx,
            extra_weight=extra_weight)

    # Apply calibration and prepare output
    if calib_wave is True:
        wave = wave + calibration[1]
    spec = spec * calibration[0]

    # Apply limb-darkening correction on atlas if need be
    if (mu != 1.0):
        limbdark_factor = np.mean(limbdarkening(wave_fts, mu=mu))
        spec_fts *= limbdark_factor
        spec_fts_dc *= limbdark_factor

    spec_fts_sel = []
    for ww in range(wave.size):
        widx = np.argmin(np.abs(wave_fts - wave[ww]))
        spec_fts_sel.append(spec_fts[widx])

    if qsdc_calib is True:
        spec /= cont_fts[0]
        spec_fts /= cont_fts[0]
        spec_fts_dc /= cont_fts[0]
        spec_fts_sel /= cont_fts[0]
        calibration[0] /= cont_fts[0]
        sunit = u.dimensionless_unscaled
    else:
        sunit = fts.sunit

    if verbose is True:
        if calib_wave is True:
            print("spectrum: wavelength calibration offset: {0} (added to input wavelengths)".format(calibration[1]))
        print("spectrum: intensity calibration offset factor: {0}".format(calibration[0]))
        if qsdc_calib is True:
            print("spectrum: STiC calibration offset factor: {0}".format(cont_fts[0]))
            plot_scale_factor = 1.0
        else:
            plot_scale_factor = 1.e-5
        profile *= calibration[0]
        fig, ax = plt.subplots()
        legend_items = ('observed profile', 'selected points', 
            'atlas profile at '+u'μ={0}'.format(mu))
        ax.plot(wave, profile/plot_scale_factor, '.')
        ax.plot(wave[wave_idx], profile[wave_idx]/plot_scale_factor, '+')
        ax.plot(wave_fts, spec_fts_dc/plot_scale_factor)
        if instrument_profile is not None:
            ax.plot(wave_fts, spec_fts/plot_scale_factor,'--')
            legend_items += ('atlas convolved with instrument profile',)
        if qsdc_calib is True:
            ax.set_ylabel('intensity relative to disc centre continuum [dimensionless]')
        else:
            ax.set_ylabel('intensity ['+r'$\times10^{-5}$'+' {0}]'.format(sunit.to_string()))
        ax.set_xlabel('wavelength [{0}]'.format(fts.wunit.to_string()))
        ax.legend(legend_items)
        ax.set_title('ISPy: calib.spectrum() results')
        plt.show()

    return wave, spec, calibration, spec_fts_sel, sunit


def limbdarkening(wave, mu=1.0, nm=False):
    """
    Return limb-darkening factor given wavelength and viewing angle
    mu=cos(theta)

    Parameters
    ----------
    wave : float or array_like
        scalar or 1D array with wavelength(s).
    mu : float, optional
        cosine of heliocentric viewing angle (default 1.0 -> disc centre)
    nm : bool, optional
        input wavelength units are nanometers (default False)

    Returns
    -------
    factor : float or array_like
        scaling factor(s) to be applied for given input wavelengths. Has as many
        elements as `wave`.

    Example
    -------
    >>> factor = limbdarkening(630.25, mu=0.7, nm=True)

    :Author:
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
