"""
Python tools to read and write 'La Palma' cubes

Merge of some routines written by Tiago Pereira (github:helita), J. de la Cruz Rodriguez,
Alex Pietrow (github:crispy), Carlos Diaz and G. Vissers (ISP/SU 2019)

"""
import numpy as np
import os
import sys


# ========================================================================
def lphead(name, verbose=True, appendFormat=False):
    """

    Args:
      name: Name of the file we want to know the header properties.
      verbose: (Default value = True)
      appendFormat: different format to ensure append operation (Default =False)

    Returns:
        header
    """
    inam = 'lphead:'

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
            nt = int(du1[1].split(',')[0])
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
        print((inam, "[dtype={0}, ndims={1}, nx={2}, ny={3}, nt={4}, nstokes={5}] -> {6}".format(
            dtype, ndims, nx, ny, nt, nstokes, os.path.basename(name))))

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
def lp_read(cube, spnw=True, ns=4, spformat='_sp', mode='r', dtype='float32', verb=False):
    """Reads a cube with La Palma format cube.

    Arguments:
        cube: filename, has to be .icube or .fcube
        spnw: Optional. Specific filename of spectral cube  OR number of wavelength steps (Default value = True)
        ns: number of stokes parameters. (Default value = 4)
        spformat: (Default value = '_sp')
        mode: (Default value = 'r')
        dtype: Type of data. Should be 'float32' for .fcubes and 'int16' for icubes. (Default value = 'float32')
        verb: Verbose mode. (Default value = False)

    Returns:
        5D cube of shape [nt,ns,nw,nx,ny]

    Examples:
        A) cube = lp_read('filename.fcube') # It will find also 'filename_sp.fcube' in the same path
        B) cube = lp_read('filename.fcube' , 8)
        C) cube = lp_read('filename.fcube' , 'filename_sp.fcube')

    Authors: Alex Pietrow (ISP/SU 2019), Carlos Diaz (ISP/SU 2019)
    """
    if type(spnw) is str:
        if ns == 4:
            nx, ny, dum, ns, dtype, ndim = lphead(cube)
            nw, nt, dum, ns, dtype, ndim = lphead(spnw)
            cube_array = np.memmap(cube, shape=(
                nt, ns, nw, ny, nx), offset=512, dtype=dtype, mode=mode)
        elif ns == 1:
            nx, ny, dum, dtype, ndim = lphead(fil0)
            nw, nt, dum, dtype, ndim = lphead(fil1)
            cube_array = np.memmap(cube, shape=(
                nt, nw, ny, nx), offset=512, dtype=dtype, mode=mode)
        else:
            raise ValueError("Stokes must be 1 or 4")

    elif type(spnw) is bool:
        cube = cube[:-6]+'{0}.fcube'
        f1 = cube.format('')
        f2 = cube.format(spformat)
        if not os.path.isfile(f2):
            raise ValueError('File '+f2+' was not found. Please '
                +'include the name of spectral file or wavelength steps.')

        if ns == 4:
            nx, ny, ndum, nstokes, dtype, dum1 = lphead(f1, verb)
            nw, nt, ndum, nstokes, dtype, dum1 = lphead(f2, verb)
            cube_array = np.memmap(f1, shape=(
                nt, ns, nw, ny, nx), offset=512, dtype=dtype, mode=mode)

    elif type(spnw) is int:
        if ns == 4:
            hheader = lphead(cube, verb)
            nt = int(hheader[2]/spnw/ns)
            nx = hheader[0]
            ny = hheader[1]
            cube_array = np.memmap(cube, shape=(
                nt, ns, spnw, ny, nx), offset=512, dtype=dtype, mode=mode)
        elif ns == 1:
            hheader = lphead(cube, verb)
            nt = int(hheader[2]/spnw)
            nx = hheader[0]
            ny = hheader[1]
            cube_array = np.memmap(cube, shape=(
                nt, spnw, ny, nx), offset=512, dtype=dtype, mode=mode)
        else:
            raise ValueError("Stokes must be 1 or 4")

    return cube_array


# ========================================================================
def mk_lpheader(image):
    """Creates header for La Palma images.

    Args:
        image: La Palma cube

    Returns:
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
    """Writes on disk a cube using LaPalma format. Backend of "lp_write"
    From https://github.com/ITA-Solar/helita/blob/master/helita/io/lp.py

    Args:
        filename: name of the file
        image: data allocated in memory
        extraheader: (Default value = '')
        dtype: (Default value = None)
        verbose: (Default value = False)
        append: (Default value = False)
    Returns
        file on disk
    """
    if not os.path.isfile(filename):
        append = False
    # use dtype from array, if none is specified
    if dtype is None:
        dtype = image.dtype
    image = image.astype(dtype)
    if append:
        # check if image sizes/types are consistent with file
        sin, t, h = lphead(filename, verbose=verbose,
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
        header = mk_lpheader(image)
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
def lp_write(cube_array, name, stokes=True, sp=False, path=''):
    """Saves cube as La Palma format cube.

    Arguments:
        cube_array: datacube in form of [t,s,w,x,y]
        name: name of file with .icube/fcube extention
        stokes: flag for if data has stokes or not. (Default value = True)
        sp: Save spectral cube of shape. (Default value = False)
        path: Filepath where file needs to be saved.(Default value = '')

    Examples:
        lp_write(cube_array, 'cube.fcube', path='fits/')

    Authors: Alex Pietrow (ISP/SU 2019), Carlos Diaz (ISP/SU 2019)
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
def lp_get(filename, index, verb=False):
    """Reads a 2D image (slice) given a known index from a La Palma cube.

    Arguments:
        filename : file to be opened. Has to be .icube or .fcube
        index    : chosen frame, where frame is t*nw*ns + s*nw + w
            t: time or scan number
            s: stokes parameter
            w: wavelength step
        verbose : Verbose model. (Default value = False)

    Returns:
        2D image

    Example:
        data = lp_get('cube.fcube', 0)

    Authors: G. Vissers (ITA UiO, 2016), A.G.M. Pietrow (2018), Carlos Diaz (ISP/SU 2019)
    """
    nx, ny, ndum, nstokes, dt, dum1 = lphead(filename, verb)
    # header offset + stepping through cube
    offset = 512 + index * nx * ny * np.dtype(dt).itemsize
    image = np.memmap(filename, dtype=dt, mode='r', shape=(nx, ny), offset=offset,
                      order='F')
    return image.T


# ========================================================================
def lp_put(filename, image, append=True, verbose=False, stokes=True):
    """Append a new cube/slice to a pre-existent La Palma cube.

    Arguments:
        filename: name of file with .icube/fcube extention
        image: datacube in form of [t,s,w,x,y]
        append: (Default value = True)

    Examples:
        lp_put('cube.fcube', cube2)

    To do:
        Insert a slice in a La Palma cube

    Authors: Carlos Diaz,  G. Vissers, A.G.M. Pietrow (ISP/SU 2019)
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

