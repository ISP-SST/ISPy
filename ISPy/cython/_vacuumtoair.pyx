import numpy as np
cimport numpy as np

cdef extern from "vacuumtoair.c":
    void vacuum_to_air(int Nlambda, double *lambda_vac, double *lambda_air)
    void air_to_vacuum(int Nlambda, double *lambda_air, double *lambda_vac)

def toair(wl):
    cdef int Nlambda;
    cdef np.ndarray[np.double_t, mode = 'c'] lambda_vacuum = np.ascontiguousarray(wl,dtype=np.double);
    Nlambda = lambda_vacuum.size
    shp = np.shape(wl)
    cdef np.ndarray[np.double_t, ndim=1, mode = 'c'] wl_air = np.empty(Nlambda, dtype=np.double, order='C')

    vacuum_to_air(Nlambda, <double*> lambda_vacuum.data, <double*> wl_air.data);

    return wl_air.reshape(shp)

def tovacuum(wl):
    cdef int Nlambda;
    cdef np.ndarray[np.double_t, mode = 'c'] lambda_air = np.ascontiguousarray(wl,dtype=np.double);
    Nlambda = lambda_air.size
    shp = np.shape(wl)
    cdef np.ndarray[np.double_t, ndim=1, mode = 'c'] wl_vacuum = np.empty(Nlambda, dtype=np.double, order='C')

    air_to_vacuum(Nlambda, <double*> lambda_air.data, <double*> wl_vacuum.data);

    return wl_vacuum.reshape(shp)
