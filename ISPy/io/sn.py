"""
Python tools to read SST data formatted as FITS files conforming 
to the SOLARNET standard.

Written by Jorrit Leenaarts (ISP/SU 2021)

"""
import astropy.io.fits as fits
import numpy as np
import os
import sys
import astropy.wcs as wcs
import datetime 

class sstdata:
    """
    class for reading and handling SST data formatted as FITS files conforming 
    to the SOLARNET standard.
    """

    ###################################################################


    def __init__(self, filename):
        """
        construct sstdata object 

        Parameters
        ----------
        filename : str
            name of the fitsfile

        Returns
        -------

        Example
        -------
        >>> from ISPy.io import sn
        >>> mydata = sn.sstdata("filename.fits")
        >>> 

        :Author: 
            Jorrit Leenaarts (ISP/SU 2021)
        """

        self.filename = filename

        if not os.path.exists(filename):
            print(filename+" does not exist.")
            return
        else:
            self.hdul = fits.open(filename)
            self.header = self.hdul[0].header
            #set up world coordinate system 
            self.w = wcs.WCS(filename)

        # what is the data order. For now we only allow for (fast to
        # slow) image cubes, not sp cubes
        self.is_imcube = (self.header['CNAME1'] == 'Spatial X')

        # main data size
        if self.is_imcube:
            self.nx = self.header['NAXIS1']
            self.ny = self.header['NAXIS2']
            self.nwav = self.header['NAXIS3']
            self.nstokes = self.header['NAXIS4']
            self.nt = self.header['NAXIS5']
        else:
            print(filename+" is not an image cube, other formats (_sp cubes) are not yet supported.")
            return

        # reference to data
        self.data = self.hdul[0].data

        # compute exact start time of the dataset
        self.dateref = self.header['DATEREF']
        self.t0 = float(self.w.wcs_pix2world(0,0,0,0,0,0)[4])
        self.dateref_start = datetime.datetime.fromisoformat(self.dateref) + datetime.timedelta(seconds = self.t0)
        
        # approximate axes, we assume that the spatial axes do not
        # change over time, the wavelength axis does not change over
        # time, and are determined from the first tuning of the first
        # scan.
        #
        # As time coordinate we set the time of the middle point of
        # the scan, starting from zero.
        #
        # If you need more precise
        # coordinates, call wcs_pix2world by hand
        self.x = np.zeros(self.nx)
        for i in range(self.nx):
            self.x[i] = self.w.wcs_pix2world(i,0,0,0,0,0)[0]

        self.y = np.zeros(self.ny)
        for i in range(self.ny):
            self.y[i] = self.w.wcs_pix2world(0,i,0,0,0,0)[1]

        self.wav = np.zeros(self.nwav)
        for i in range(self.nwav):
            self.wav[i] = self.w.wcs_pix2world(0,0,i,0,0,0)[2]

        self.t = np.zeros(self.nt)
        for i in range(self.nt):
            self.t[i] = self.w.wcs_pix2world(0,0,self.nwav//2,0,i,0)[4]
        self.t -= self.t[0]
