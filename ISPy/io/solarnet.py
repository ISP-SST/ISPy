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
    """Reads the data of a 'SOLARNET' cube.

    Arguments:
        filename: name of the data cube

    Returns:
        1D wavelength array in nm.

    Authors: Carlos Diaz (ISP/SU 2019)
    """
    return fits.getdata(filename, ext=0)


# ========================================================================
def write(filename, d):
    """Reads the data as a normal fits (without proper SOLARNET header)

    Arguments:
        filename: name of the data cube
        d: data on memory

    Authors: Carlos Diaz (ISP/SU 2019)
    """
    io = fits.PrimaryHDU(d)
    io.writeto(filename, overwrite=True)
    return


# ========================================================================
def get_wav(filename):
    """Reads the wavelength information of a 'SOLARNET' cube.

    Arguments:
        filename: name of the data cube
        - It is assumed that spectral sampling does not change over time

    Returns:
        1D wavelength array in nm.

    Example:
        wav_array = get_wav(filename)

    Authors: Carlos Diaz (ISP/SU 2019)
    """
    io = fits.open(filename)
    # Wavelength information for all wavelength points in the first time frame
    wav_output = io[1].data[0][0][0,:,0,0,2]
    return wav_output


# ========================================================================
def seconds2string(number):
    """Converts time in seconds to 'HH:MM:SS' format

    Arguments:
        number: value in seconds

    Returns:
        Time in 'HH:MM:SS' format

    Authors: Carlos Diaz (ISP/SU 2019)
    """
    hour = number//3600.
    minute = (number%3600.)//60.
    seconds = (number-(hour*3600.+minute*60))/1.
    string_output = '{0:02}:{1:02}:{2:09.6f}'.format(int(hour),int(minute),seconds)
    return string_output


# ========================================================================
def get_time(filename, fulltime=False, utc=False):
    """Reads the time information of a 'SOLARNET' cube.

    Arguments:
        filename: name of the data cube (*im.fits only)
        fulltime: information at each wavelength point (Default value = False)
        utc: the output is given in 'HH:MM:SS' format (Default value = False)

    Returns:
        fulltime=False : 1D array with time information at middle wavelength point.
        fulltime=True : 2D array with time information at each wavelength point.

    Example:
        data_time = get_time(filename, fulltime=False, utc=True)

    Authors: Carlos Diaz (ISP/SU 2019)
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
    """Reads the coordinates of the corners to use them with imshow/extent

    Arguments:
        filename: name of the data cube
        timeFrame: information at a given time frame (Default value = 0)

    Returns:
        1D array with solar coordinates in arcsec of the corners.

    Example:
        extent = get_extent(filename)
        imshow(data, extent=extent)

    Authors: Carlos Diaz (ISP/SU 2019)
    """
    io = fits.open(filename)
    extent_output = [io[1].data[0][0][timeFrame,0,0,0,0],io[1].data[0][0][timeFrame,0,1,1,0],
            io[1].data[0][0][timeFrame,0,0,0,1],io[1].data[0][0][timeFrame,0,1,1,1]]

    return extent_output

# ========================================================================
def get_coord(filename, pix_x,pix_y,timeFrame=0):
    """Converts pixels values to solar coordinates

    Arguments:
        filename: name of the data cube
        pix_x, pix_y : location to convert
        timeFrame: information at a given time frame (Default value = 0)

    Returns:
        Solar coordinates in arcsec

    Example:
        [x_output, y_output] = get_coord(filename, pix_x,pix_y)

    Authors: Carlos Diaz (ISP/SU 2019)
    """
    io = fits.open(filename)

    yp = [0, io[0].data.shape[3]]
    xp = [0, io[0].data.shape[4]]

    fxp = [io[1].data[0][0][timeFrame,0,0,0,0], io[1].data[0][0][timeFrame,0,1,1,0]]
    fyp = [io[1].data[0][0][timeFrame,0,0,0,1], io[1].data[0][0][timeFrame,0,1,1,1]]

    x_output = np.interp(pix_x, xp, fxp)
    y_output = np.interp(pix_y, yp, fyp) 

    return [x_output, y_output]
