import numpy as np

def cder(x, y):

    ny, nx, nstokes, nlam = y.shape[:]
    yp = np.zeros((ny, nx, nlam), dtype='float32')

    odx = x[1]-x[0]; ody = (y[:,:,0,1] - y[:,:,0,0]) / odx
    yp[:,:,0] = ody

    for ii in range(1,nlam-1):
        dx = x[ii+1] - x[ii]
        dy = (y[:,:,0,ii+1] - y[:,:,0,ii]) / dx

        yp[:,:,ii] = (odx * dy + dx * ody) / (dx + odx)

        odx = dx; ody = dy

    yp[:,:,-1] = ody    
    return yp


class line:
    def __init__(self, cw=0):

        self.larm = 4.668645048281451e-13
        
        if(cw == 8542):
            self.j1 = 2.5; self.j2 = 1.5; self.g1 = 1.2; self.g2 = 1.33; self.cw = 8542.091
        elif(cw == 6301):
            self.j1 = 2.0; self.j2 = 2.0; self.g1 = 1.84; self.g2 = 1.50; self.cw = 6301.4995
        elif(cw == 6302):
            self.j1 = 1.0; self.j2 = 0.0; self.g1 = 2.49; self.g2 = 0.0; self.cw = 6302.4931
        elif(cw == 8468):
            self.j1 = 1.0; self.j2 = 1.0; self.g1 = 2.50; self.g2 = 2.49; self.cw = 8468.4059
        else:
            print("line::init: Warning, line not implemented")
            self.j1 = 0.0; self.j2 = 0.0; self.g1 = 0.0; self.g2 = 0.0; self.cw = 0.0
            return

        j1 = self.j1; j2 = self.j2; g1 = self.g1; g2 = self.g2
        
        d = j1 * (j1 + 1.0) - j2 * (j2 + 1.0);
        self.geff = 0.5 * (g1 + g2) + 0.25 * (g1 - g2) * d;
        ss = j1 * (j1 + 1.0) + j2 * (j2 + 1.0);
        dd = j1 * (j1 + 1.0) - j2 * (j2 + 1.0);
        gd = g1 - g2;
        self.Gg = (self.geff * self.geff) - (0.0125  * gd * gd * (16.0 * ss - 7.0 * dd * dd - 4.0));

        print("line::init: cw={0}, geff={1}, Gg={2}".format(self.cw, self.geff, self.Gg))

    def update(j1, j2, g1, g2, cw):
        """
        This method updates the atomic data, useful for lines that are not defined in the constructor
        Example:
            lin = line(cw=0)
            lin.update(2.5, 1.5, 1.2, 1.33, 8542.091)
        """
        self.j1 = j1; self.j2 = j2; self.g1 = g1; self.g2 = g2; self.cw = cw
        
        d = j1 * (j1 + 1.0) - j2 * (j2 + 1.0);
        self.geff = 0.5 * (g1 + g2) + 0.25 * (g1 - g2) * d;
        ss = j1 * (j1 + 1.0) + j2 * (j2 + 1.0);
        dd = j1 * (j1 + 1.0) - j2 * (j2 + 1.0);
        gd = g1 - g2;
        self.Gg = (self.geff * self.geff) - (0.0125  * gd * gd * (16.0 * ss - 7.0 * dd * dd - 4.0));

        print("line::init: cw={0}, geff={1}, Gg={2}".format(self.cw, self.geff, self.Gg))


def getBlos(w, d, line, beta = 0.0):
    """
    Computes the line of sight component of the magnetic field vector
    Arguments:
        w: 1D numpy array with the wavelength offsets on the observation (can be relative to line-center)
        d: 4D numpy array (ny, nx, nstokes, nwav) with the observations
        line: object from line class (also included) with the atomic data of the transition
        beta: (optional) a regularization factor that selects low-norm solutions. Use wisely.

    Coded by J. de la Cruz Rodriguez (ISP-SU 2018)
    """
    c = -line.larm * line.cw**2 * line.geff; cc = c*c
    der = cder(w, d)

    Blos = (c*d[:,:,3,:] * der).sum(axis=2) / (cc*(der**2).sum(axis=2) + beta)
    return Blos


def getBlosProf(w, d, line, beta = 0.0):

    c = -line.larm * line.cw**2 * line.geff; cc = c*c
    der = cder(w, d)

    Blos = (c*d[:,:,3,:] * der).sum(axis=2) / (cc*(der**2).sum(axis=2) + beta)
    return Blos, c*der*Blos


def getBlosMask(w, d, line, mask, beta = 0.0):
    """
    Computes the line of sight component of the magnetic field vector, using selected wavelength points of the observation
    Arguments:
        w: 1D numpy array with the wavelength offsets on the observation (can be relative to line-center)
        d: 4D numpy array (ny, nx, nstokes, nwav) with the observations
        line: object from line class (also included) with the atomic data of the transition
        mask: a tuple/numpy array with the wavlength indexes that should be used to compute Blong
        beta: (optional) a regularization factor that selects low-norm solutions. Use wisely.

    Coded by J. de la Cruz Rodriguez (ISP-SU 2018)
    """
    c = -line.larm * line.cw**2 * line.geff; cc = c*c
    der = cder(w, d)

    Blos = (c*d[:,:,3,mask] * der[:,:,mask]).sum(axis=2) / (cc*(der[:,:,mask]**2).sum(axis=2) + beta)
    return Blos
    
def getBhor(w, d, lin, beta = 0.0, vdop = 0.05):
    """
    Computes the horizontal component of the magnetic field vector
    Arguments:
        w: 1D numpy array with the wavelength offsets on the observation (can be relative to line-center)
        d: 4D numpy array (ny, nx, nstokes, nwav) with the observations
        line: object from line class (also included) with the atomic data of the transition
        beta: (optional) a regularization factor that selects low-norm solutions. Use wisely.
        vdep: (optional) typical Doppler width of the line. Wavelength points inside +- vdop from line center will be excluded
    Coded by J. de la Cruz Rodriguez (ISP-SU 2018)
    """
    c = 0.75 * (lin.larm * lin.cw**2)**2 * lin.Gg; cc=c*c
    der = cder(w,d)

    for ii in range(len(w)):
        if(np.abs(w[ii]) >= vdop): scl = 1./w[ii]
        else: scl = 0.0
        der[:,:,ii] *= scl

        
    Q_term = (d[:,:,1,:] * der).sum(axis=2) 
    U_term = (d[:,:,2,:] * der).sum(axis=2) 

    Bhor = ((c*(Q_term**2 + U_term**2)**0.5) / (cc * (der*der).sum(axis=2) + beta))**0.5
    return Bhor
