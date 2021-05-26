"""
Python tools to read and write 'La Palma' cubes

Merge of some routines written by Tiago Pereira (github:helita), J. de la Cruz Rodriguez,
Alex Pietrow (github:crispy), Carlos Diaz and G. Vissers (ISP/SU 2019)

"""
import numpy as np
import os
import sys


# ========================================================================
def head(name, verbose=False, appendFormat=False):
    """
    Get the header of a legacy 'La Palma' cube

    Parameters
    ----------
    name : str
        name of the file 
    verbose : bool, optional
        print out the header information (Default value = True)
    appendFormat : bool, optional
        different format to ensure append operation (Default =False)

    Returns
    -------
    header : tuple
        header information in order (nx, ny, nt, nstokes, dtype, ndims)

    Example
    -------
    >>> h = head('crispex_3950_2016-09-19T09:28:36_scans=11-117_time-corrected_im.fcube')
    ('head:', '[dtype=float32, ndims=3, nx=1734, ny=1240, nt=3317, nstokes=1] -> crispex_3950_2016-09-19T09:28:36_scans=11-117_time-corrected_im.fcube')
    >>> nx, ny, nt, ns, dtype, ndims = head('crispex_3950_2016-09-19T09:28:36_scans=11-117_time-corrected_im.fcube', verbose=False)
    >>> nx, ny, ns
    (1734, 1240, 1)
    """

    inam = 'head:'

    # Open file
    datfil = open(name, 'rb')

    # get header and extract dimensions
    head = (np.fromfile(datfil, dtype=np.dtype('a512'), count=1))[0]

    dum = head.decode("utf-8").split()
    datfil.close()

    ndims = 0
    dtype = 0
    nx = 0
    ny = 0
    nt = 0
    nstokes = 1

    for it in dum:
        du1 = it.split("=")
        if("dims" in du1[0]):
            ndims = int((du1[1].split(','))[0])
        elif("datatype" in du1[0]):
            dtype = int(du1[1].split(',')[0])
        elif("nx" in du1[0]):
            nx = int(du1[1].split(',')[0])
        elif("ny" in du1[0]):
            ny = int(du1[1].split(',')[0])
        elif("nt" in du1[0]):
            try:
                nt = int(du1[1].split(',')[0])
            except:
                pass # (integer label)
        elif("stokes" in du1[0]):
            du2 = du1[1].split(']')
            nstokes = int(np.size(du2[0].split(',')))

    if(dtype == 1):
        dtype = np.dtype('b')
    elif(dtype == 2):
        dtype = np.dtype('h')
    elif(dtype == 3):
        dtype = np.dtype('i')
    elif(dtype == 4):
        dtype = np.dtype('f')
    elif(dtype == 5):
        dtype = np.dtype('d')
    else:
        print((inam, 'Warning, dtype={0} not supported!'.format(dtype)))

    if(verbose):
        print(inam, "[dtype={0}, ndims={1}, nx={2}, ny={3}, nt={4}, nstokes={5}] -> {6}".format(
            dtype, ndims, nx, ny, nt, nstokes, os.path.basename(name)))

    if appendFormat is True:
        # read header and convert to string
        h = np.fromfile(name, dtype='uint8', count=512)
        header = ''
        for s in h[h > 0]:
            header += chr(s)
        # start reading at 'datatype'
        hd = header[header.lower().find('datatype'):]
        hd = hd.split(':')[0].replace(',', ' ').split()
        # Types:   uint8  int16 int32 float32
        typelist = ['u1', 'i2', 'i4', 'f4']
        # extract datatype
        try:
            dtype = typelist[int(hd[0].split('=')[1]) - 1]
        except:
            print(header)
            raise IOError('getheader: datatype invalid or missing')
        # extract endianness
        try:
            if hd[-1].split('=')[0].lower() != 'endian':
                raise IndexError()
            endian = hd[-1].split('=')[1]
        except IndexError:
            print(header)
            raise IOError('getheader: endianess missing.')
        if endian.lower() == 'l':
            dtype = '<' + dtype
        else:
            dtype = '>' + dtype
        # extract dims
        try:
            if hd[2].split('=')[0].lower() != 'dims':
                raise IndexError()
            dims = int(hd[2].split('=')[1])
            if dims not in [2, 3]:
                raise ValueError('Invalid dims=%i (must be 2 or 3)' % dims)
        except IndexError:
            print(header)
            raise IOError('getheader: dims invalid or missing.')
        try:
            if hd[3].split('=')[0].lower() != 'nx':
                raise IndexError()
            nx = int(hd[3].split('=')[1])
        except:
            print(header)
            raise IOError('getheader: nx invalid or missing.')
        try:
            if hd[4].split('=')[0].lower() != 'ny':
                raise IndexError()
            ny = int(hd[4].split('=')[1])
        except:
            print(header)
            raise IOError('getheader: ny invalid or missing.')
        if dims == 3:
            try:
                if hd[5].split('=')[0].lower() != 'nt':
                    raise IndexError()
                nt = int(hd[5].split('=')[1])
            except:
                print(header)
                raise IOError('getheader: nt invalid or missing.')
            shape = (nx, ny, nt)
        else:
            shape = (nx, ny)
        return [shape, dtype, header]

    return nx, ny, nt, nstokes, dtype, ndims


# ========================================================================
def read(cube, spnw=None, spformat='_sp', verb=False):
    """
    Read the full cube from a La Palma format file

    Parameters
    ----------
    cube : str
        filename, has to be .icube or .fcube
    spnw : str or int, optional 
        Specific filename of spectral cube OR number of wavelength steps (Default value = None)
    spformat : str, optional
        filename identifier for the spectral cube (Default value = '_sp')
    verb : bool, optional
        Verbose mode. (Default value = False)

    Returns
    -------
    cube_array: ndarray
        5D cube of shape [nt,ns,nw,nx,ny]

    Examples
    --------
    >>> from ISPy.io import lapalma as lp
    >>> cube_a = lp.read('filename.fcube') # It searches for 'filename_sp.fcube' in the same path
    >>> cube_b = lp.read('filename.fcube' , 8)
    >>> cube_c = lp.read('filename.fcube' , 'filename_sp.fcube')

    :Authors: 
        Alex Pietrow (ISP/SU 2019), Carlos Diaz (ISP/SU 2019)
    """
    if spnw is None:
        cube_format = cube[:-6]+'{0}.'+cube[-5:]
        f1 = cube_format.format('')
        f2 = cube_format.format(spformat)
        if not os.path.isfile(f2):
            raise ValueError('File '+f2+' was not found. Please '
                +'include the name of spectral file or wavelength steps.')
        nx, ny, ndum, ns, dtype, ndim = head(f1, False)
        nw, nt, ndum, ns, dtype, ndim = head(f2, False)
        cube_array = np.memmap(f1, shape=(
            nt, ns, nw, ny, nx), offset=512, dtype=dtype, mode='r')


    elif type(spnw) is str:
        f2 = str(spnw)
        if not os.path.isfile(f2):
            raise ValueError('File '+f2+' was not found. Please '
                +'include the name of spectral file or wavelength steps.')
        nx, ny, ndum, ns, dtype, ndim = head(f1, False)
        nw, nt, ndum, ns, dtype, ndim = head(f2, False)
        cube_array = np.memmap(f1, shape=(
            nt, ns, nw, ny, nx), offset=512, dtype=dtype, mode='r')


    elif type(spnw) is int:
        nw = int(spnw)
        nx, ny, hh2, ns, dtype, ndim = head(cube, False)
        nt = int(hh2/nw/ns)
        cube_array = np.memmap(cube, shape=(
            nt, ns, nw, ny, nx), offset=512, dtype=dtype, mode='r')


    if(verb):
        print("[dtype={0}, nx={2}, ny={3}, nt={4}, nstokes={5}, nwav={1}] -> {6}".format(
            dtype, nw, nx, ny, nt, ns, os.path.basename(cube)))

    return cube_array


# ========================================================================
def mk_header(image):
    """
    Create a La Palma format header of an image array 

    Parameters
    ----------
    image : ndarray
        2D or 3D image array in La Palma ordering (nx, ny, nt)

    Returns
    -------
    header : str
        header of the cube
    """
    from struct import pack
    ss = image.shape
    # only 2D or 3D arrays
    if len(ss) not in [2, 3]:
        raise IndexError(
            'make_header: input array must be 2D or 3D, got %iD' % len(ss))
    dtypes = {'int8': ['(byte)', 1], 'int16': ['(integer)', 2], 'int32': [
        '(long)', 3], 'float32': ['(float)', 4]}
    if str(image.dtype) not in dtypes:
        raise ValueError('make_header: array type' +
                         ' %s not supported, must be one of %s' % (image.dtype, list(dtypes.keys())))
    sdt = dtypes[str(image.dtype)]
    header = ' datatype=%s %s, dims=%i, nx=%i, ny=%i' % (
        sdt[1], sdt[0], len(ss), ss[0], ss[1])
    if len(ss) == 3:
        header += ', nt=%i' % (ss[2])
    # endianess
    if pack('@h', 1) == pack('<h', 1):
        header += ', endian=l'
    else:
        header += ', endian=b'
    return header


# ========================================================================
def writeto(filename, image, extraheader='', dtype=None, verbose=False,
            append=False):
    """
    Submodule of "write". It writes a cube to disk in LaPalma format.
    Partially from https://github.com/ITA-Solar/helita/blob/master/helita/io/lp.py

    Parameters
    ----------
    filename : str
        name of the file
    image : ndarray
        data allocated in memory
    extraheader : str, optional
        extra header information to append to standard header (Default value = '')
    dtype : str, optional
        data type of the image (Default value = None)
    verbose : bool, optional
        verbose mode (Default value = False)
    append : bool, optional
        append `image` to existing file (Default value = False)

    Returns
    -------
    NoneType

    Examples
    --------
    writeto('path/cube.fcube', image, append=True)
    """
    
    if not os.path.isfile(filename):
        append = False
    # use dtype from array, if none is specified
    if dtype is None:
        dtype = image.dtype
    image = image.astype(dtype)
    if append:
        # check if image sizes/types are consistent with file
        sin, t, h = head(filename, verbose=verbose,
                           appendFormat=True)  # getheader(filename)
        if sin[:2] != image.shape[:2]:
            raise IOError('writeto: trying to write' +
                          ' %s images, but %s has %s images!' %
                          (repr(image.shape[:2]), filename, repr(sin[:2])))
        if np.dtype(t) != image.dtype:
            raise IOError('writeto: trying to write' +
                          ' %s type images, but %s nas %s images' %
                          (image.dtype, filename, np.dtype(t)))
        # add the nt of current image to the header
        hloc = h.lower().find('nt=')
        new_nt = str(sin[-1] + image.shape[-1])
        header = h[:hloc + 3] + new_nt + h[hloc + 3 + len(str(sin[-1])):]
    else:
        header = mk_header(image)
    if extraheader:
        header += ' : ' + extraheader
    # convert string to [unsigned] byte array
    hh = np.zeros(512, dtype='uint8')
    for i, ss in enumerate(header):
        hh[i] = ord(ss)
    # write header to file
    file_arr = np.memmap(filename, dtype='uint8',
                         mode=append and 'r+' or 'w+', shape=(512,))
    file_arr[:512] = hh[:]

    del file_arr
    # offset if appending
    apoff = append and np.prod(sin) * image.dtype.itemsize or 0
    # write array to file
    file_arr = np.memmap(filename, dtype=dtype, mode='r+',
                         order='F', offset=512 + apoff, shape=image.shape)
    file_arr[:] = image[:]

    del file_arr
    if verbose:
        if append:
            print(('Appended %s %s array into %s.' %
                   (image.shape, dtype, filename)))
        else:
            print(('Wrote %s, %s array of shape %s' %
                   (filename, dtype, image.shape)))
    return


# ========================================================================
def write(cube_array, name, stokes=True, sp=False, path=''):
    """
    Write a data cube in La Palma format to disc 

    Parameters
    ----------
    cube_array : ndarray
        datacube in form of [t,s,w,x,y]
    name : str
        name of file with .icube/fcube extention
    stokes : bool, optional
        flag for if data has stokes or not. (Default value = True)
    sp : bool, optional
        Save spectral cube of shape. (Default value = False)
    path : str, optional
        Filepath where file needs to be saved.(Default value = '')

    Examples
    --------
    >>> from ISPy.io import lapalma as lp
    >>> lp.write(cube_array, 'cube.fcube', path='fits/')

    :Authors: 
        Alex Pietrow (ISP/SU 2019), Carlos Diaz (ISP/SU 2019)
    """

    # Reshaping to save it in the right format:
    intensity = np.moveaxis(cube_array.astype(np.float32), [
                            0, 1, 2, 3, 4], [1, 0, -1, -2, 2])

    # 'write_buf' function is included
    if not stokes:
        nt, nx, ny, nw = intensity.shape
        ax = [(1, 2, 0, 3), (3, 0, 2, 1)]
        rs = [(nx, ny, nt * nw), (nw, nt, ny * nx)]
        extrahd = ''
    else:
        ns, nt, nx, ny, nw = intensity.shape
        ax = [(2, 3, 1, 0, 4), (4, 1, 3, 2, 0)]
        rs = [(nx, ny, nt * ns * nw), (nw, nt, ny * nx * ns)]
        extrahd = ', stokes=[I,Q,U,V], ns=4'

    # this is the image cube:
    im = np.transpose(intensity, axes=ax[0])
    im = im.reshape(rs[0])
    # this is the spectral cube
    if sp:
        sp = np.transpose(intensity, axes=ax[1])
        sp = sp.reshape(rs[1])
        writeto(path+name+'_sp.fcube', sp, extraheader=extrahd)

    writeto(path+name+'.fcube', im, extraheader=extrahd)
    return


# ========================================================================
def get(filename, index, verb=False):
    """
    Read a 2D image (slice) given a known index from a La Palma cube.

    Parameters
    ----------
    filename : str
        file to be opened. Has to be .icube or .fcube
    index : int
        chosen frame, where frame is t*nw*ns + s*nw + w
        t: time or scan number
        s: stokes parameter
        w: wavelength step
    verbose : bool, optional
        Verbose model. (Default value = False)

    Returns
    -------
    image : ndarray
        2D image slice

    Examples
    --------
    >>> image = lp_get('cube.fcube', 0)

    :Authors:
        G. Vissers (ITA UiO, 2016), A.G.M. Pietrow (2018), Carlos Diaz (ISP/SU 2019)
    """
    nx, ny, ndum, nstokes, dt, dum1 = head(filename, verb)
    # header offset + stepping through cube
    offset = 512 + index * nx * ny * np.dtype(dt).itemsize
    image = np.memmap(filename, dtype=dt, mode='r', shape=(nx, ny), offset=offset,
                      order='F')
    return image.T


# ========================================================================
def put(filename, image, append=True, verbose=False, stokes=True):
    """
    Append a new cube/slice to a pre-existent La Palma cube.

    Parameters
    ----------
    filename : str
        name of file with .icube/fcube extention
    image : ndarray
        datacube in form of [t,s,w,x,y]
    append : bool, optional
        append `image` to an existing file (Default value = True)
    verbose : bool, optional
        verbose mode (default False)
    stokes : bool, optional
        data has Stokes parameters (default True)

    Examples
    --------
    >>> lp_put('cube.fcube', cube2)

    To do:
        Insert a slice in a La Palma cube

    :Authors: 
        Carlos Diaz,  G. Vissers, A.G.M. Pietrow (ISP/SU 2019)
    """

    # Reshaping to save it in the right format:
    intensity = np.moveaxis(image.astype(np.float32), [
                            0, 1, 2, 3, 4], [1, 0, -1, -2, 2])

    # 'write_buf' function is included
    if not stokes:
        nt, nx, ny, nw = intensity.shape
        ax = [(1, 2, 0, 3), (3, 0, 2, 1)]
        rs = [(nx, ny, nt * nw), (nw, nt, ny * nx)]
        extrahd = ''
    else:
        ns, nt, nx, ny, nw = intensity.shape
        ax = [(2, 3, 1, 0, 4), (4, 1, 3, 2, 0)]
        rs = [(nx, ny, nt * ns * nw), (nw, nt, ny * nx * ns)]
        extrahd = ', stokes=[I,Q,U,V], ns=4'

    # this is the image cube:
    im = np.transpose(intensity, axes=ax[0])
    im = im.reshape(rs[0])

    writeto(filename, im, verbose=verbose, append=append)
    return

