import numpy as np

# ========================================================================
def get_projection(u,v):
    """It outputs the scalar projection using the inner product of the vectors.
    
    Parameters
    ----------
    u, v : ndarray
        input vectors
    
    Returns
    -------
    float
        Scalar projection
    """

    return np.inner(u,v)/np.inner(v,v)

# ========================================================================
def estimate_crosstalk(stokes_map,stokes_toclean, stokes_removefrom, nameoutput=None, interactive=True, 
    npoints=5, verbose=True, sizeave=3, smoothvalue=100):
    """Given the input cube it outputs an estimation of the crosstalk between them 
    for one time frame.
    
    Parameters
    ----------
    stokes_map : ndarray
        Data cube for one time frame
    stokes_toclean : int
        Stokes index to clean (usually 1 or 2)
    stokes_removefrom : int
        Stokes index to clean (usually 3)
    nameoutput : None, optional (default: None)
        Name of the output file with the field-dependent crosstalk and figures
    interactive : bool, optional (default: True)
        Interactive method where the user selects points in the FOV.
    npoints : int, optional (default: 5)
        Number of points to be selected
    verbose : bool, optional (default: True)
        It prints the average crosstalk calculated with both methods
    sizeave : int, optional (default: 3)
        Window size to average after clicking
    smoothvalue : int, optional (default: 100)
        Smoothing width of the kernel after extrapolation
    
    Returns
    -------
    ndarray
        Field-dependent crosstalk if interactive is True or a the mean value calculated by the automatic method if False
       
    Example
    -------
    >>> estimate_crosstalk(data[0,:], stokes_toclean=1, stokes_removefrom=3, nameoutput='Q_test')
    
    :Authors: 
        Carlos Diaz (ISP/SU 2019)

    """
    stokes_label = ['I','Q','U','V']
    coefmap = np.zeros((stokes_map.shape[2],stokes_map.shape[3]),dtype=np.float32)
    weights = np.zeros((stokes_map.shape[2],stokes_map.shape[3]),dtype=np.float32)
    
    for ii in range(stokes_map.shape[2]):
        for jj in range(stokes_map.shape[3]):
            coefmap[ii,jj] = get_projection(stokes_map[stokes_toclean,:,ii,jj]/stokes_map[0,:,ii,jj],
                                             stokes_map[stokes_removefrom,:,ii,jj]/stokes_map[0,:,ii,jj])
            weights[ii,jj] = np.max(np.abs(stokes_map[stokes_removefrom,:,ii,jj]))**2. 

    mean_xtalk = np.sum(coefmap*weights)/np.sum(weights)
    if verbose is True: print('Average crosstalk (automatic method): {0:2.2f}%'.format(mean_xtalk*100.))

    if interactive is True:
        
        import matplotlib.pyplot as plt
        fig = plt.figure()
        maxi = np.min([np.percentile(np.abs(coefmap),95),0.5])
        plt.imshow(coefmap,vmax=maxi,vmin=-maxi,cmap='seismic',origin='lower')
        plt.colorbar(); plt.title('Crosstalk {} -> {}'.format(stokes_label[stokes_removefrom],stokes_label[stokes_toclean]))
        if nameoutput is not None: plt.savefig(nameoutput+'_crosstalk_map.pdf', bbox_inches='tight')

        print('Click on {} different places'.format(npoints))
        points = plt.ginput(npoints, timeout=30, show_clicks=True)
        values = []
        for ii in range(len(points)):
            mean_points = np.mean( coefmap[int(points[ii][1])-sizeave:int(points[ii][1])+sizeave+1 ,
                                    int(points[ii][0])-sizeave:int(points[ii][0])+sizeave+1] )
            values.append(mean_points)
        points = np.array(points)
        values = np.array(values)

        plt.close(fig)

        # Interpolation and smoothing
        from scipy.interpolate import griddata
        grid_x, grid_y = np.meshgrid(np.arange(coefmap.shape[1]), np.arange(coefmap.shape[0]))
        grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
        newmap = grid_z0
        from scipy import ndimage
        mean_xtalk = ndimage.gaussian_filter(newmap,smoothvalue)
        
        if nameoutput is not None: np.save(nameoutput+'.npy',mean_xtalk)
        if nameoutput is not None: print('Field-dependent crosstalk saved in '+ nameoutput+'.npy')

        if nameoutput is not None: 
            fig = plt.figure()
            plt.imshow(mean_xtalk,vmax=maxi,vmin=-maxi,cmap='seismic',origin='lower')
            plt.colorbar(); plt.title('Crosstalk {} -> {}'.format(stokes_label[stokes_removefrom],stokes_label[stokes_toclean]))
            plt.savefig(nameoutput+'_crosstalk_estimation.pdf', bbox_inches='tight')
            plt.close(fig)

    if verbose is True: print('Average crosstalk (interactive method): {0:2.2f}%'.format(np.mean(mean_xtalk)*100.))

    return mean_xtalk



# ========================================================================
if __name__ == "__main__":

    print('[INFO] Reading dataset')
    from ISPy.io import solarnet as sl

    data_final = 'filename_stokes_corrected_im.fits'
    data = sl.read(data_final)

    estimate_crosstalk(data[0,:],stokes_toclean=1,stokes_removefrom=3,nameoutput='Q_test')

