import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ====================================================================
def BTAZI2BQBU(Bt,azi):
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
def BQBU2BTAZI(BQ,BU):
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
def BTAZI2BQBU_cube(model_Bho, model_azi):
    """ Transformation from transverse field and azimuth to BQ and BU
    for a 2D cube array.
    """
    model_BQ = np.ones_like(model_Bho)
    model_BU = np.ones_like(model_Bho)
    for x in range(model_Bho.shape[0]):
        for y in range(model_Bho.shape[1]):
            for ii in range(model_Bho.shape[2]):
                model_BQ[x,y,ii], model_BU[x,y,ii] = BTAZI2BQBU(model_Bho[x,y,ii],model_azi[x,y,ii])
    return model_BQ, model_BU

# ====================================================================
def BQBU2BTAZI_cube(model_BQ, model_BU):
    """Transformation from BQ and BU to transverse field and azimuth
    for a 2D cube array.
    """
    model_Bho = np.ones_like(model_BQ)
    model_Bazi = np.ones_like(model_BQ)
    for x in range(model_Bho.shape[0]):
        for y in range(model_Bho.shape[1]):
            for ii in range(model_Bho.shape[2]):
                model_Bazi[x,y,ii], model_Bho[x,y,ii] = BQBU2BTAZI(model_BQ[x,y,ii],model_BU[x,y,ii])
    return model_Bho, model_Bazi



if __name__ == '__main__':
    # TEST
    Bt = 100.
    azi = 80.* (np.pi/180.)
    print(Bt,azi)
    BTAZI2BQBU(Bt,azi)
    BQBU2BTAZI(BQ,BU)







