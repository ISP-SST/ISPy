import numpy as np
import matplotlib.pyplot as pl
import h5py
import time
from readout_fn import *
from ipdb import set_trace as stop

'''
program to show the observed and best fitted profiles interactively.
Click on the inverted maps to see the observed and synthetic profiles.
Inputs: observed and synthetic profiles, inverted maps, and wavelength range file
'''
def xyplot(ix,iy):
	pl.close(2)
	fig, ax = pl.subplots(figsize=(10,6), nrows=2, ncols=2,num = 'OBSERVED & SYNTHETIC PROFILES')
	ax = ax.flatten()
	ix = int(np.ceil(ix))
	iy = int(np.ceil(iy))
	for i in range(0,4):
		ax[i].plot(wav-10830.,obspro[ix,iy,:,i])
		#ax[i].plot(wav-10830.,synpro[ix,iy,i,:])
		ax[i].plot(wav-10830.,synpro1[ix,iy,i,:])
		#ax[i].plot(wav-10830.,synpro2[ix,iy,i,:])
		if i ==0:ax[0].text(-6.5, 0.5, 'pixels: ('+str(iy)+','+str(ix)+')')

	pl.tight_layout()		
	pl.show()
############################
	
def onclick(event):
	#pl.close(1)
	global ix, iy
	ix, iy = event.xdata, event.ydata
	print("x-pos:",np.ceil(ix)," y-pos:", np.ceil(iy))
	
	xyplot(iy,ix)
	#if len(coords) == 5:
		#fig.canvas.mpl_disconnect(cid)
		#pl.close()
	#return coords
##########################	

if __name__ == "__main__":
	global obspro,synpro,synpro1,synpro2
	
	invf = 'outputs/comp1/xybin_0403_00.h5'			#inverted maps
	wavf = 'wavelength_2bin_trim.txt'			#wavelenght scale	
	obsf = 'observations/spatially_binned_2pix.h5'		#observed Stokes profiles

	wav = np.loadtxt(wavf)					#read wavelength file
	f1 = h5py.File(obsf,'r')				#read observed profiles
	prof = f1['stokes']

	npix,nlambda,stks = f1['stokes'].shape
	#inv_ch,inv_ph, synpro = readout_1c(obsf,invf)
	#inv_ch,inv_ph, synpro1, tau, fa,chi = readout_1c(obsf,125,226,invf)
	inv_ch, synpro1,chi = readout_1c_ch(obsf,125,226,invf)	#get synthetic profiles and inverted maps

	nx,ny,stk,lmb = synpro1.shape
	
	obspro  = np.reshape(prof,[nx,ny,nlambda,stks])

	mod =  inv_ch		#inverted maps
	
	fig, ax = pl.subplots(figsize=(7,5), nrows=3,ncols=2,num = 'INVERTED MAPS')
	ax = ax.flatten()
	chlabel = ['tau','v','Bx','By','Bz','chi2','beta','deltav','ff']
	vminv = ['-30','-500', '-500','-500']
	vmaxv = ['30','500', '500','500']
	for i in range(0,6):
		
		if i == 0 or i >4:
			im = ax[i].imshow(mod[:,:,i], origin='lower', cmap=pl.cm.bwr)
		else:
			im = ax[i].imshow(mod[:,:,i], origin='lower', cmap=pl.cm.bwr,vmin = vminv[i-1], vmax = vmaxv[i-1])
		if i == 5: im = ax[i].imshow(chi, origin='lower', cmap=pl.cm.bwr)
		fig.colorbar(im, ax=ax[i],orientation="horizontal",label=chlabel[i])

	cid = fig.canvas.mpl_connect('button_press_event', onclick)
	pl.tight_layout()	
	pl.show()
	#fig.canvas.mpl_disconnect(cid)
