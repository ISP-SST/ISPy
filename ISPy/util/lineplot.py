#!/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import scipy.io as sc
import numpy as np
import ISPy.spec.atlas as atlas
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
import ISPy.spec.crisp as fpi
import ISPy.spec.chromis as ch
import sys
import astropy.io.fits as f
from ipdb import set_trace as stop


def find_nearest(array, array2, k=1):
    array = np.asarray(array)
    arr = np.zeros_like(array2)
    for i in range(len(array2)):
        arr[i] = np.int(np.argsort(abs(array - array2[i]))[:k])
    return arr


def SST_scan_time(nlambda, nstates, nframes, nprefilter=1, fps=35):
    """
    Calculate the length of one scan based on observational parameters.
    
    
    Parameters
    ----------
    nlambda : float
        Amount of wavelength points in scan all lines combined
        Multiply number of points by 4 if line has polerimetry.
    nstates : float
        Number of polarization states. Should be 1 or 4.
    nframes : float
        Number of frames per state
    nprefilter : float, optional
        Number of prefilters used
    fps : float, optional
        Frames per second. Default = 35
        
    Returns
    -------
    t
       Scan duration in seconds
    
    Example
    --------
       Duration for observation of a line with 28 points + continuum
       with 6 states and polarization.
       
       from ISPy.util import lineplot as lp
       t = lp.SST_scan_time(29,4,6,2,35)
    
    :Authors:
        Alex Pietrow (ISP-SU 2021), based on private communication between Alex and Pit.
    
    """
    
    if nprefilter > 1:
        
        #2 frames lost for every shift in lambda
        nframes = ( nlambda * ( nstates + 2 ) ) + 10 * nprefilter
        t       = nframes/fps
        
    else:
        
        #2 frames lost for every shift in lambda
        nframes = nlambda * ( nstates + 2 )
        t       = nframes/fps

    return t
    

    
def plotline(line,scan,name=None,center=0,nframes=8,fps=35., instrument='CRISP', width=2):
    """
    Visualize observing program per line.
    
    
    Parameters
    ----------
    line : float
        line center in Angstrom
    scan : array
        Numpy array with values observed, centered around line center in Angstrom
    name : str, optional
        filename and title. If set to zero, file will not be shown.
    center : float, optional
        Force line center to value provided in line.
        Defealt = False, which will center the line at the minimum value.
    nframes : float, optional
        Number of frames per state. Default = 8
    fps : float, optional
        frames per second. Default = 35
    instrument : str
        Set instrument to 'CRISP' or 'CHROMIS'
    width : float
        Half spectral range in Angstrom.
        
    Returns
    -------

    
    Example
    --------

    from ISPy.util import lineplot as lp
    import numpy as np

    #Crisp line
    line = 6563 #A
    dl = 62/2
    scan = np.append(np.append(np.arange(-1*dl*65, -1*dl*20, dl*5), np.arange(-1*dl*20,dl*20,dl)),np.arange(dl*20, dl*70, dl*5))
    lp.plotline(line,scan/1000.,str(line)+'.png', width=2.5)

    #Chromis line
    line = 3934
    dl = 120/2
    scan = np.arange(-1*dl*20,dl*20,dl)
    lp.plotline(line,scan/1000.,str(line)+'.png',instrument='CHROMIS', width=2)

    
    :Authors:
        Alex Pietrow (ISP-SU 2021)
    
    """

    #get solar atlas spectrum
    s = atlas.atlas()
    lambda_atlas,intensity_atlas,c = s.get(line-width,  line+width, cgs=True)
    
    if center == 0:
        w_atlas_central = np.argmin(intensity_atlas) #find line center
        cw = lambda_atlas[np.argmin(intensity_atlas)]
    else:
        cen = len(lambda_atlas)/2
        w_atlas_central = np.argmin(intensity_atlas[cen-50:cen+50]) + cen -50 #find line center
        cw = lambda_atlas[np.argmin(intensity_atlas[cen-50:cen+50])]
        
    w_atlas         = lambda_atlas - lambda_atlas[w_atlas_central] #relative center
    dw_atlas        = w_atlas[1] - w_atlas[0] #wavelength steps
    


    #make central. 0 must be central pixel.
    #Cut array edges if this is not the case

    w0 = (len(w_atlas) - 2* np.where(w_atlas == 0)[0][0])
    if w0 < 0:
        w_atlas         = w_atlas[-w0:]
        intensity_atlas = intensity_atlas[-w0:]
    if w0 > 0:
        w_atlas         = w_atlas[:-w0]
        intensity_atlas = intensity_atlas[:-w0]

    #find nearest points to atlas. The resolution is good enough.
    points = find_nearest(w_atlas, scan).astype(int)
    
    #pick instrument
    
    if instrument == 'CRISP':
        instrument_profile = fpi.crisp(cw) #get profile
        transmission_profile     = instrument_profile.dual_fpi(w_atlas) #make transmission prof
    else:
        print("chromis::read_reflectivity")
        transmission_profile     = ch.dual_fpi(lambda_atlas,w0=0)
        
        
    transmission_profile     = transmission_profile / np.sum(np.abs(transmission_profile)) #normalize

    #fft does not work with small numbers, so need to make larger
    intensity_atlas_convolved= fftconvolve(intensity_atlas/np.mean(intensity_atlas), transmission_profile, mode='same')*np.mean(intensity_atlas)
    
    #plot
    plt.clf()
    plt.plot(w_atlas, intensity_atlas_convolved)
    plt.scatter(w_atlas[points], intensity_atlas_convolved[points])
    plt.title(str(line)+' Å\n t='+str(int(np.round(len(points)*nframes/fps)))+' s')
    plt.xlabel(r'Wavelength Å')
    plt.ylabel(r'Intensity [erg/s/cm$^2$/sr/Å]')
    plt.tight_layout()
    plt.plot()
    if name:
        plt.savefig(name)
    else:
        plt.show()
    
    
    


