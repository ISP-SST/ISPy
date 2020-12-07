import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ====================================================================
def fromBTAZI2BQBU(Bt,azi):
    """Transformation from transverse field and azimuth to BQ and BU
    according to https://arxiv.org/abs/1904.03714

    Parameters
    ----------
    Bt, azi : float
        Values to be transformed
    """
    By = Bt*np.sin(azi)
    Bx = Bt*np.cos(azi)
    BQ = np.sign(Bx**2.-By**2.)*np.sqrt(np.abs(Bx**2.-By**2.))
    BU = np.sign(Bx*By)*np.sqrt(np.abs(Bx*By))
    return BQ,BU

# ====================================================================
def fromBQBU2BTAZI(BQ,BU):
    """Transformation from BQ and BU to transverse field and azimuth
    according to https://arxiv.org/abs/1904.03714

    Parameters
    ----------
    BQ, BU : float
        Values to be transformed
    """
    A = BQ**2.
    B = -BU**4.
    s4 = np.nan_to_num( np.sqrt(-A + np.sqrt(A**2 - 4*B))/np.sqrt(2) )
    if BQ < 0 and BU < 0:
        Bx_r = -s4
        By_r = -BU**2./Bx_r
    if BQ > 0 and BU < 0:
        By_r = s4
        Bx_r = -BU**2./By_r
    if BQ < 0 and BU > 0:
        Bx_r = s4
        By_r = BU**2./Bx_r
    if BQ >= 0 and BU >= 0:
        By_r = s4
        Bx_r = BU**2./By_r
    azi_r = np.arctan2(By_r,Bx_r)
    if azi_r <0: azi_r = azi_r+np.pi
    azi_r = ( azi_r )/ (np.pi/180.)
    Bt_r = np.sqrt(By_r**2.+Bx_r**2.)
    return azi_r, Bt_r

# ====================================================================
def fromBTAZI2BQBU_cube(model_Bho, model_azi):
    """ Transformation from transverse field and azimuth to BQ and BU
    for a 2D cube array.
    """
    model_BQ = np.ones_like(model_Bho)
    model_BU = np.ones_like(model_Bho)
    for x in range(model_Bho.shape[0]):
        for y in range(model_Bho.shape[1]):
            for ii in range(model_Bho.shape[2]):
                model_BQ[x,y,ii], model_BU[x,y,ii] = fromBTAZI2BQBU(model_Bho[x,y,ii],model_azi[x,y,ii])
    return model_BQ, model_BU

# ====================================================================
def fromBQBU2BTAZI_cube(model_BQ, model_BU):
    """Transformation from BQ and BU to transverse field and azimuth
    for a 2D cube array.
    """
    model_Bho = np.ones_like(model_BQ)
    model_Bazi = np.ones_like(model_BQ)
    for x in range(model_Bho.shape[0]):
        for y in range(model_Bho.shape[1]):
            for ii in range(model_Bho.shape[2]):
                model_Bazi[x,y,ii], model_Bho[x,y,ii] = fromBQBU2BTAZI(model_BQ[x,y,ii],model_BU[x,y,ii])
    return model_Bho, model_Bazi


# ====================================================================
from matplotlib.colors import Normalize
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)
    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# ====================================================================
import matplotlib.colors as mcolors
def make_colormap(seq):
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

# ====================================================================
# Some definitions of colormaps
c = mcolors.ColorConverter().to_rgb
vcolormap = make_colormap([c('darkred'),c('oldlace'),0.5, c('lightcyan'),c('midnightblue')])
phimap = make_colormap([c('white'), c('tomato'), 0.33, c('tomato'), c('deepskyblue'), 0.66, c('deepskyblue'),c('white')])


if __name__ == '__main__':
    # TEST
    Bt = 100.
    azi = 80.* (np.pi/180.)
    print(Bt,azi)
    fromBTAZI2BQBU(Bt,azi)
    fromBQBU2BTAZI(BQ,BU)







