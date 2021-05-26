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
