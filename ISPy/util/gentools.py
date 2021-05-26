import numpy as np


# ==================================================================================
def rotate_cube(cube, angle_index):
    """Rotate a cube with dim=[x,y,z] a multiple of 90 degrees in plane x,y

    Parameters
    ----------
    cube : 3D ndarray
        Array with values to rotate
    angle_index : int
        Times to rotate 90 degrees the cube

    :Authors: 
        Carlos Diaz (ISP/SU 2020)

    """
    # Rotate a cube with dim=[x,y,z] a multiple of 90 degrees in plane x,y
    nnqq = cube.shape[2]
    ncube = np.copy(cube)
    for ii in range(nnqq):
        ncube[:, :, ii] = np.rot90(cube[:, :, ii], angle_index)
    return ncube



# ==================================================================================
def findclose(value, array):
    """Finds the index of the closest value in array

    Parameters
    ----------
    value : float
        value we want to find
    array : ndarray
        array with values where we want to search

    :Authors: 
        Carlos Diaz (ISP/SU 2020)

    """
    return np.argmin(np.abs(value - array))


# ==================================================================================
def findindex(array1, array2):
    """Outputs an array with the indices of the closes element in array2 for each 
    element in array1

    Parameters
    ----------
    array1 : ndarray
        Array with values that we want to find
    array2 : ndarray
        Array with values  where we want to search

    :Authors: 
        Carlos Diaz (ISP/SU 2020)

    """
    index_array = np.ones_like(array1)
    for kk in array1:
        index_array[kk] = findclose(kk, array2)
    return index_array

