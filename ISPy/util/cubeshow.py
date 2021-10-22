import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


###
# animate_cube()
##
def animate_cube(cube_array, name=None, show=True, cut=True, mn=0, sd=0, interval=75, cmap='gray', fps=20):
    '''
    Animates a python cube for quick visualisation and saves mp4.
    
    Parameters
    ----------
    cube_array : 3D array[t,x,y]
        The array to be animated.
    name    : bool, optional
        name of file to save, must end with .mp4. Will not save if not used. Default = None
    show    : bool, optional
        Wheather or not to display animation. Default = True.
    cut     : int, optional
        trims pixels off of the images edge to remove edge detector effects. Default = True as 0 returns empty array.
    mn      : float, optional
        mean value to used for contrast. Default=0
    sd      : float, optional
        Std value to used for contrast. Default=0
    interval: int, optional
        #of ms between each frame. Default=75
    cmap    : str, optional
        colormap used for animation. Default='gray'
    fps     : float, optional
        frames per second. Default = 20
   
        
    Returns
    -------
        Video file if name is given.

    Example
    --------
    from ISPy.util import cubeshow as cs
    
    a = np.random.random([90,100,100])
    cs.animate_cube(a, 'video.mp4',cut=20,interval=100)
    
    :Authors:
        Alex Pietrow (ISP-SU 2021)
    
    '''
    if show:
        fig = plt.figure()
        std = np.std(cube_array[0])
        mean = np.mean(cube_array[0])
        if mn==sd and mn==0:
            img = plt.imshow(cube_array[0][cut:-cut, cut:-cut], animated=True, vmax=mean+3*std, vmin=mean-3*std, cmap=cmap)
        else:
            img = plt.imshow(cube_array[0][cut:-cut, cut:-cut], animated=True, vmax=mn+3*sd, vmin=mn-3*sd, cmap=cmap)
        
        def updatefig(i):
            img.set_array(cube_array[i][cut:-cut, cut:-cut])
            return img,
        
        ani = animation.FuncAnimation(fig, updatefig, frames=cube_array.shape[0],                                  interval=interval, blit=True)
        plt.colorbar()
        plt.show()
    
    if name:
        print("Saving mp4 now...")
        
        fig = plt.figure()
        std = np.std(cube_array[0])
        mean = np.mean(cube_array[0])
        if mn==sd and mn==0:
            img = plt.imshow(cube_array[0][cut:-cut, cut:-cut], animated=True, vmax=mean+3*std, vmin=mean-3*std, cmap=cmap)
        else:
            img = plt.imshow(cube_array[0][cut:-cut, cut:-cut], animated=True, vmax=mn+3*sd, vmin=mn-3*sd, cmap=cmap)
        
        def updatefig(i):
            img.set_array(cube_array[i][cut:-cut, cut:-cut])
            return img,
    
        
        ani = animation.FuncAnimation(fig, updatefig, frames=cube_array.shape[0], interval=interval, blit=True)
        plt.colorbar()
        
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='ISPy'), bitrate=1800)
        ani.save(name, writer=writer)
        print('Done!')
        

def plotall(cube_array, fignum=np.array([4,6]), title='Title', save=False):
    '''
    Plot a 3D cube into a subplot array
    
    Parameters
    ----------
    cube_array : 3D array[t,x,y]
        The array to be animated.
    fignum     : 2D array
        Number of subplots. Default = np.array([4,6])
    title      : str, optional
        Title of plot, defealt='Title'
    save       : bool, optional
        Saves output to file named after title. Default = False
        
    Returns
    -------

    Example
    --------
       a = np.random.random([9,100,100])
       plotall(a, fignum=np.array([3,3]), title='test',save=1)
    
    :Authors:
        Alex Pietrow (ISP-SU 2021)
    
    '''
    fig, axs = plt.subplots(fignum[0], fignum[1], figsize=(15, 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)
    
    axs = axs.ravel()
    plt.title(title)
    
    for i in range(cube_array.shape[0]):
        axs[i].imshow(cube_array[i])
        axs[i].set_title('frame '+str(i))
        
    plt.show()
