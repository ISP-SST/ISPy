import numpy as np

def avg1D(xdata,ydata,nbin):
	'''
	Inputs:
		xdata: 1D arrray
		ydata: 1D array
		nbin: number of bins
	returns: 
		xbin, ybin: rebinned 1D arrays (x & y) 	
	'''
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
	'''
	Inputs:
		data: 2D arrray
		nbin: number of bins
	returns: 
		rebin: rebinned 2D array
	'''
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
