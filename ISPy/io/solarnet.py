"""
Python tools to read and write 'SOLARNET' cubes

Written by Carlos Diaz (ISP/SU 2019)

"""
import astropy.io.fits as fits
import numpy as np
import os
import sys

# ========================================================================
def read(filename):
    """
    Read the data of a 'SOLARNET' FITS file 

    Parameters
    ----------
    filename : str
        name of the data cube

    Returns
    -------
    data : ndarray
        multidimensional data array 

    Example
    -------
    >>> from ISPy.io import solarnet
    >>> data = solarnet.read('filename.fits')
    >>> data.shape
    (30, 4, 41, 1914, 206)

    :Author: 
        Carlos Diaz Baso (ISP/SU 2019)
    """
    return fits.getdata(filename, ext=0)


# ========================================================================
def write(filename, d):
    """
    Write the data as a standard FITS file (without proper SOLARNET header)

    Parameters
    ----------
    filename : str
        name of the data cube
    d : ndarray
        data in memory

    Example
    -------
    >>> from ISPy.io import solarnet
    >>> solarnet.write('filename.fits', data)

    :Author: 
        Carlos Diaz Baso (ISP/SU 2019)
    """
    io = fits.PrimaryHDU(d)
    io.writeto(filename, overwrite=True)
    return


# ========================================================================
def get_wav(filename):
    """
    Read the wavelength information of a 'SOLARNET' FITS file

    Parameters
    ----------
    filename : str
        name of the data cube

    Returns
    -------
    wav_output : array_like
        1D wavelength array. It is assumed that spectral sampling does not
        change over time.

    Example
    -------
    >>> from ISPy.io import solarnet
    >>> wav_array = solarnet.get_wav('filename.fits')

    :Authors:
        Carlos Diaz Baso (ISP/SU 2019)
    """
    io = fits.open(filename)
    # Wavelength information for all wavelength points in the first time frame
    wav_output = io[1].data[0][0][0,:,0,0,2]
    return wav_output


# ========================================================================
def seconds2string(number):
    """
    Convert time in seconds to 'HH:MM:SS' format

    Parameters
    -----------
    number : float
        value in seconds

    Returns
    -------
    time_string : str
        time in 'HH:MM:SS' format

    Example
    -------
    >>> from ISPy.io import solarnet
    >>> solarnet.seconds2string(63729.3)
    '17:42:09.300000'

    :Author:
        Carlos Diaz Baso (ISP/SU 2019)
    """
    hour = number//3600.
    minute = (number%3600.)//60.
    seconds = (number-(hour*3600.+minute*60))/1.
    string_output = '{0:02}:{1:02}:{2:09.6f}'.format(int(hour),int(minute),seconds)
    return string_output


# ========================================================================
def get_time(filename, fulltime=False, utc=False):
    """
    Reads the time information of a 'SOLARNET' FITS file

    Parameters
    ----------
    filename : str
        name of the data cube
    fulltime : bool, optional
        information at each wavelength point (Default value = False)
    utc : bool, optional
        the output is given in 'HH:MM:SS' format (Default value = False)

    Returns
    -------
    time_output : array_like
        array with time information. If fulltime=False, a 1D array with time at
        middle wavelength tuning; if fulltime=True, a 2D array with
        tuning-dependent time information.

    Example
    -------
    >>> from ISPy.io import solarnet
    >>> data_time = solarnet.get_time(filename, fulltime=False, utc=True)

    :Author: 
        Carlos Diaz Baso (ISP/SU 2019)
    """
    io = fits.open(filename)

    wav_len =  io[1].data[0][0].shape[1]
    middle_wav = wav_len//2

    if fulltime is True:
        time_output = io[1].data[0][0][:,:,0,0,3]

    else:
        time_output = io[1].data[0][0][:,middle_wav,0,0,3]

    if utc is True:
        time_output_ = np.array([seconds2string(number) for number in time_output.reshape(time_output.size)])
        time_output = time_output_.reshape(time_output.shape)

    return time_output


# ========================================================================
def get_extent(filename, timeFrame=0):
    """
    Read the coordinates of the corners to use them with imshow/extent

    Parameters
    filename : str
        name of the data cube
    timeFrame : int
        selected frame to get the coordinates for (Default value = 0)

    Returns
    -------
    extent_output : array_like
        1D array with solar coordinates in arcsec of the corners.

    Example
    -------
    >>> from ISPy.io import solarnet
    >>> extent = solarnet.get_extent(filename)
    >>> plt.imshow(data, extent=extent)

    :Author: 
        Carlos Diaz Baso (ISP/SU 2019)
    """
    io = fits.open(filename)
    extent_output = [io[1].data[0][0][timeFrame,0,0,0,0],io[1].data[0][0][timeFrame,0,1,1,0],
            io[1].data[0][0][timeFrame,0,0,0,1],io[1].data[0][0][timeFrame,0,1,1,1]]

    return extent_output

# ========================================================================
def get_coord(filename, pix_x,pix_y,timeFrame=0):
    """
    Converts pixel values to solar coordinates

    Parameters
    ----------
    filename : str
        name of the data cube
    pix_x, pix_y : int
        pixel location to convert
    timeFrame : int
        selected frame to the coordinates for (Default value = 0)

    Returns
    -------
    xy_output : list
        solar coordinates in arcsec

    Example
    -------
    >>> from ISPy.io import solarnet
    >>> [x_output, y_output] = solarnet.get_coord(filename, pix_x,pix_y)

    :Author: 
        Carlos Diaz Baso (ISP/SU 2019)
    """
    io = fits.open(filename)

    yp = [0, io[0].data.shape[3]]
    xp = [0, io[0].data.shape[4]]

    fxp = [io[1].data[0][0][timeFrame,0,0,0,0], io[1].data[0][0][timeFrame,0,1,1,0]]
    fyp = [io[1].data[0][0][timeFrame,0,0,0,1], io[1].data[0][0][timeFrame,0,1,1,1]]

    x_output = np.interp(pix_x, xp, fxp)
    y_output = np.interp(pix_y, yp, fyp) 

    return [x_output, y_output]

def get_partial(filename, timeFrame=0, stokes_sel=None, wav_sel=None,
                return_xy=True, squeeze=True, memmap=False):
    """
    Read a single time frame from a SOLARNET FITS cube without loading everything.

    Assumes FITS axis order:
      NAXIS1=x, NAXIS2=y, NAXIS3=lambda, NAXIS4=stokes, NAXIS5=time

    Parameters
    ----------
    filename : str
        SOLARNET cube filename (PrimaryHDU contains data)
    timeFrame : int, optional
        Time index to read (0-based). Default: 0
    stokes_sel : None, int, slice, array-like of int, optional
        Selection for Stokes axis after reading (e.g. 0, slice(None), [0,3]).
        If None: keep all.
    wav_sel : None, int, slice, array-like of int, optional
        Selection for wavelength axis after reading.
        If None: keep all.
    return_xy : bool, optional
        If True, also return x and y coordinate axes in arcsec derived from SOLARNET metadata.
    squeeze : bool, optional
        If True, squeeze singleton dimensions (e.g. stokes if NAXIS4==1).
    memmap : bool, optional
        Passed to fits.open for header access. Default False since you are doing raw reads anyway.

    Returns
    -------
    (x, y, wav, stokes, cube) if return_xy is True
    (wav, stokes, cube)      if return_xy is False

    Notes
    -----
    cube is returned in numpy order:
      (nstokes, nlambda, ny, nx)
    i.e. cube[s, l, y, x]

    This reads from the raw FITS byte stream using the PrimaryHDU data offset.

    from ISPy.io import solarnet
    x, y, wav, stokes, cube = solarnet.get_partial(fn, timeFrame=0)

    :Author: 
        AGM Pietrow (AIP 2026)
    """
    # --- Read header + data offset ---
    with fits.open(filename, memmap=memmap) as hdul:
        hdu0 = hdul[0]
        hdr = hdu0.header

        nx = int(hdr["NAXIS1"])
        ny = int(hdr["NAXIS2"])
        nl = int(hdr["NAXIS3"])
        ns = int(hdr.get("NAXIS4", 1))
        nt = int(hdr.get("NAXIS5", 1))

        offset = hdu0._data_offset

    if timeFrame < 0 or timeFrame >= nt:
        raise IndexError(f"timeFrame={timeFrame} out of range [0, {nt-1}]")

    # --- Bytes to read for one time frame (all stokes + all lambda + full image) ---
    itemsize = 4  # float32
    bytes_per_frame = nx * ny * nl * ns * itemsize

    # --- Seek to the requested time frame ---
    frame_offset = offset + timeFrame * bytes_per_frame

    with open(filename, "rb") as f:
        f.seek(frame_offset)
        raw = f.read(bytes_per_frame)

    if len(raw) != bytes_per_frame:
        raise RuntimeError(
            f"Frame {timeFrame} incomplete: got {len(raw)} of {bytes_per_frame} bytes"
        )

    # FITS stores big-endian; float32 in your cube
    arr = np.frombuffer(raw, dtype=">f4")

    # Reshape to (stokes, lambda, y, x)
    cube = arr.reshape(ns, nl, ny, nx).astype(np.float32, copy=False)

    # --- Build wavelength + stokes coordinate arrays ---
    wav = get_wav(filename)  # expects SOLARNET ext=1 metadata layout (your existing function)
    if wav is None or len(wav) != nl:
        # Defensive fallback if metadata is missing/mismatched
        wav = np.arange(nl, dtype=float)

    stokes = np.arange(ns, dtype=int)

    # --- Apply selections (after reading) ---
    if stokes_sel is not None:
        cube = cube[stokes_sel, ...]
        stokes = stokes[stokes_sel] if np.ndim(stokes_sel) != 0 else np.array([int(stokes_sel)])

    if wav_sel is not None:
        cube = cube[:, wav_sel, ...]
        wav = wav[wav_sel] if np.ndim(wav_sel) != 0 else np.array([wav[int(wav_sel)]])

    # --- Optionally return x/y axes in arcsec ---
    if return_xy:
        # extent = [xmin, xmax, ymin, ymax] for the selected time frame
        extent = get_extent(filename, timeFrame=timeFrame)
        xmin, xmax, ymin, ymax = extent

        # 1D coordinate axes corresponding to pixel centers
        x = np.linspace(xmin, xmax, nx, dtype=np.float64)
        y = np.linspace(ymin, ymax, ny, dtype=np.float64)

        if squeeze:
            cube_out = np.squeeze(cube)
        else:
            cube_out = cube

        return x, y, wav, stokes, cube_out

    else:
        if squeeze:
            cube_out = np.squeeze(cube)
        else:
            cube_out = cube

        return wav, stokes, cube_out

