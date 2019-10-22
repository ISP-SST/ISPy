import numpy as np
import matplotlib.pyplot as pl
from scipy import signal
from tqdm import tqdm
import h5py

def binavg1D(xdata,ydata,nbin):
	#input: 1D arrray (x & y), number of bins (nbin)
	#output: 1D array of nbinned data
	
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


def binavg2D(data,nbin):
	#input: 2D array, number of bins (nbin, same for x & y direction) 
	#output: binned 2D array

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
