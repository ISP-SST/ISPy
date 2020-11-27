import numpy as np

def field_test(amp=1.0, nx=50, ny=50):
    """
    Creates a test magnetic field 2D array
    
    Parameters
    ----------
    amp : float
        Maximum strength of the magnetic field
    nx, ny : integer
        Size of the map in each dimension

    """
    x01=0.0; y01=0.0; x02=0.0; y02=0.5; sigma=1.0; amp=1.0
    bz0 = np.zeros((nx,ny))
    for j in range(ny):
        for i in range(nx):
            bz0[i,j]= amp*((1.0/((((i-(nx/2.0))/(nx/2.0))-x01)**2+(((j-(ny/2.0))/(ny/2.0))-y01)**2+1.0**2)**(3.0/2.0))-
                      (1.0/((((i-(nx/2.0))/(nx/2.0))-x02)**2+(((j-(ny/2.0))/(ny/2.0))-y02)**2+1.0**2)**(3.0/2.0)))
    return bz0



def potential_extrapolation(Bz, zz=[0.0], pixel=[0.1,0.1]):
    """
    Computes a potential extrapolation from the observed vertical field.
    It uses a fast implementation in the Fourier space.

    Parameters
    ----------
    Bz : ndarray
        2D array of dimensions (ny, nx) in Gauss
    zz : ndarray
        1D array of dimensions (nz) in Mm
    pixel : ndarray
        1D array of dimensions (2) with the pixel size in arcsec
    
    Returns
    -------
    ndarray
       3D array of dimensions (nx,ny,nz) with the magnetic field vector
    
    Example
    --------
    >>> a = readimage('Bz.fits')
    >>> Bvector = potential_extrapolation(Bz, zz=[0.0,1.0], pixel=[0.1,0.1])
    
    :Authors:
        Ported from the IDL /ssw/packages/nlfff/idl/FFF.pro (by Yuhong Fan) by Carlos Diaz (ISP-SU 2020)
        and simplified to the vertical case.
    
    """

    # Simplifications to the pure vertical case:
    b0=0.; pangle=0.; cmd=0.; lat=0.; alpha = 0.; bc = 0.; lc = 0.
    axx = 1.0; axy = 0.; axz = 0.; ayx = 0.
    ayy = 1.0; ayz = 0.; azx = 0.; azz = 1.0

    Nx1, Ny1 = Bz.shape
    nz = len(zz)
    dcterm = np.sum(Bz)/(float(Nx1)*float(Ny1))  # Flux imbalance
    Bz = Bz - dcterm

    cxx = axx ; cxy = ayx; cyx = axy; cyy = ayy
    fa = np.fft.fft2(Bz)/(Bz.shape[0]*Bz.shape[1]) # Normalization IDL fft
    fa[0,0] = 0.0   # make sure net flux is zero
    
    kxi = np.array([2*np.pi/Nx1*np.roll(np.arange(Nx1)- int((Nx1-1)/2),int(-(Nx1-1)/2))]*Ny1)
    kyi = 2*np.pi/Ny1*np.roll(np.arange(Ny1)- int((Ny1-1)/2),int(-(Ny1-1)/2)).T
    kyi = np.matmul( np.expand_dims(kyi,axis=-1), np.ones((1,Nx1)))
    dxi,dyi = pixel[0], pixel[1]
    dxi = abs((149e3/206265.0)*dxi)
    dyi = abs((149e3/206265.0)*dyi) # pixel size in Mm.
    kxi = kxi/dxi ; kyi = kyi/dyi # radians per Mm
    kx = cxx*kxi + cyx*kyi ; ky = cxy*kxi + cyy*kyi ; kz2 = (kx**2+ky**2) 
    k = np.sqrt(kz2-alpha**2.)
    kl2 = np.zeros_like(kz2) + kz2 * 1j
    kl2[0,0] = 1.0  # [0,0] is singular, do not divide by zero
    nx, ny = Bz.shape

    # Computing the vector field:
    iphihat0 = fa/kl2
    nz = len(zz);
    eps = 1e-10
    B = np.zeros((nx,ny,nz,3))
    for iz in range(0,nz):
        iphihat = iphihat0*np.exp(-k*zz[iz])
        fbx = (k*kx-alpha*ky)*iphihat
        fby = (k*ky+alpha*kx)*iphihat
        fbz = np.complex(0.,1.)*(kz2)*iphihat
        B[:,:,iz,0] = np.flipud(np.fliplr(np.real(np.fft.fft2(fbx,axes=(-1, -2))) +eps))
        B[:,:,iz,1] = np.flipud(np.fliplr(np.real(np.fft.fft2(fby,axes=(-1, -2))) +eps))
        B[:,:,iz,2] = np.flipud(np.fliplr(np.real(np.fft.fft2(fbz,axes=(-1, -2))) +eps))

    # Flux balance back
    B[:,:,:,2] = B[:,:,:,2] + dcterm
    return B




def get_acute_angle(azimuth,reference,value=90.):
    """
    It outputs the closest azimuth to the reference (adding 180 degrees)

    Parameters
    ----------
    azimuth : ndarray
        2D array of dimensions (ny, nx) in degrees
    reference : float
        Reference angle to rotate the azimuth
    value: float
        Threshold to be applied to find the closest azimuth

    Returns
    -------
    ndarray
        2D array of dimensions (nx,ny) with the closest azimuth to the reference
    
    """
    index = np.abs(azimuth-reference) > value
    new_azimuth = azimuth.copy()
    new_azimuth[index] += 180.
    return new_azimuth


def smooth_azimuth(azimap,value):
    """
    Smoothing of an azimuth map with values in the range [0, pi]

    Parameters
    ----------
    azimap : ndarray
        magnetic field azimuth of shape (1,), (nx) or (ny, nx)

    value: float
        Standard deviation for Gaussian kernel


    """
    from scipy.ndimage import gaussian_filter
    termA = np.sin(azimap*2.)
    termB = np.cos(azimap*2.)
    termA = gaussian_filter(termA,value)
    termB = gaussian_filter(termB,value)
    total = np.arctan2(termA,termB)
    total[total <0] = total[total <0] % (2*np.pi)
    return total/2.



if __name__ == '__main__':

    import matplotlib.pyplot as plt

    test1 = True

    if test1 == True:

        # Potential extrapolation
        #++++++++++++++++++++++++
        Bz = field_test()
        B = potential_extrapolation(Bz, zz=[0.0], pixel=[0.1,0.1])
        print(B.shape)

        label = ['Bx', 'By', 'Bz']
        plt.close()
        height = 0
        fg, ax = plt.subplots(1,3)
        for ii in range(3):
            ax[ii].set_title(label[ii])
            ax[ii].imshow(B[:,:,height,ii])
        plt.show()

        # Disambiguation
        #+++++++++++++++
        height = 0
        Bx, By, Bz = B[:,:,height,0], B[:,:,height,1], B[:,:,height,2]
        potential_azimuth = np.arctan2(Bx,By)*180./np.pi
        
        azimuth_test = (potential_azimuth+360.) % 180. # From inversion
        offset = 0. #Extra offset
        azimuth_test = azimuth_test + offset

        new_azimuth = get_acute_angle(azimuth_test,potential_azimuth)


        plt.imshow(Bz)
        ineach = 2
        x = np.arange(Bx.shape[0])
        y = np.arange(Bx.shape[1])
        X, Y = np.meshgrid(x, y)
        flowx = +np.sin(new_azimuth*np.pi/180.)
        flowy = -np.cos(new_azimuth*np.pi/180.)
        flox = +np.sin(azimuth_test*np.pi/180.)
        floy = -np.cos(azimuth_test*np.pi/180.)
        steps = (slice(None,None,ineach),slice(None,None,ineach))
        Q = plt.quiver(X[steps], Y[steps], flowx[steps], flowy[steps],color='r')
        Q = plt.quiver(X[steps], Y[steps], flox[steps], floy[steps],color='k')
        plt.show()



