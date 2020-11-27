
def align(a, b):        
    """
    Compute the shifts (dy,dx) needed to align image b into a

    Parameters
    ----------
    a : ndarray
        2D numpy array of dimensions (ny, nx)
    b : ndarray
        2D numpy array of dimensions (ny, nx)

    Returns
    -------
    dxy : tuple
        the (y,x) shifts in pixel units.
    
    Example
    -------
    >>> a = readimage('bla.fits')
    >>> b = readimage('ble.fits')
    >>> dy,dx = align(a,b)

    :Author:
        Jaime de la Cruz Rodriguez (ISP/SU 2015), ported from IDL
    """
    if(a.shape[0] != b.shape[0] or a.shape[1] != b.shape[1]):
        print("align: ERROR, both images must have the same size")
        return(0.0,0.0)
    
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)

    cc = np.roll(np.roll(np.real(np.fft.ifft2(fa.conjugate() * fb)), -int(fa.shape[0]//2), axis=0), -int(fa.shape[1]//2), axis=1)
    
    mm = np.argmax(cc)
    xy = ( mm // fa.shape[1], mm % fa.shape[1])

    cc = cc[xy[0]-1:xy[0]+2, xy[1]-1:xy[1]+2]
    y = 2.0*cc[1,1]
    x = (cc[1,0]-cc[1,2])/(cc[1,2]+cc[1,0]-y)*0.5
    y = (cc[0,1]-cc[2,1])/(cc[2,1]+cc[0,1]-y)*0.5

    x += xy[1] - fa.shape[1]//2
    y += xy[0] - fa.shape[0]//2

    return(y,x)
