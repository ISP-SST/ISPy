import numpy as np

def avg1D(xdata,ydata,nbin):
	"""
        Rebin 1D arrays into `nbin` bins by averaging

        Parameters
        ----------
        xdata : array_like
            1D array with abcissa
        ydata : array_like
            1D array with ordinate
        nbin : int
            number of bins

	Returns
        -------
        xbin, ybin : array_like
            rebinned 1D arrays

        Example
        -------
        >>> xbin, ybin = avg1D(xdata, ydata, 5)

        :Author:
            Rahul Yadav (ISP/SU 2019)
        """
	nx = len(ydata)
	binv = int((nx)/nbin)

	if (nbin/nx > 1):
		print('Error: Bin size > the array size!')
	else:
		ybin = np.zeros(binv,dtype = np.float64)
		xbin = np.zeros(binv,dtype = np.float64)
		for i in range(binv):
			ybin[i] = np.mean(ydata[i*nbin:i*nbin+nbin])
			xbin[i] = np.mean(xdata[i*nbin:i*nbin+nbin])
		#print('binned array',binned)
		return xbin, ybin


def avg2D(data,nbin):
	"""
        Rebin a 2D array into `nbin` bins by averaging

        Parameters
        ----------
        data : ndarray
            a 2D array
        nbin : int
            number of bins
	
        Returns
        -------
        rebin : ndarray
            rebinned 2D array

        Example
        -------
        >>> databin = avg2D(data, 10)

        :Author:
            Rahul Yadav (ISP/SU 2019)
        """

	nx,ny = data.shape
	npx = int(nx/nbin)
	npy = int(ny/nbin)
	rebin = np.zeros((npx, npy),dtype = np.float64) 
	if (nbin/nx > 1):
		print('Error: Bin size > the array size!')
	else:
		for i in range(npx):
			for j in range(npy):
				rebin[i,j] = np.mean(data[i*nbin:i*nbin+nbin,j*nbin:j*nbin+nbin])

		#print('binned array',binned)
	return rebin
