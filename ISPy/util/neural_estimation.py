import copy
import numpy as np
import os
import random
import sys
import time
import warnings
from contextlib import redirect_stdout
import ipdb

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
# To deactivate warnings: https://github.com/tensorflow/tensorflow/issues/7778
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import keras.backend.tensorflow_backend as ktf
    from keras.callbacks import Callback, ModelCheckpoint
    from keras.layers import Input, Conv2D, add, Activation
    from keras.layers import Lambda
    from keras.layers.advanced_activations import ELU
    from keras.layers.merge import concatenate
    from keras.models import Model
    from keras.optimizers import Adam
    from keras.regularizers import l2
    from keras.utils.data_utils import Sequence
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True))

except ImportError:
    print('Please install keras and tensorflow to continue.')
    sys.exit()

try:
    import sparsetools as sp
except ImportError:
    print('Please install sparsetools to continue.')
    sys.exit()


from ISPy.util import gentools


# ==================================================================================
class DataGenerator(Sequence):
    """Generates data for training a neural network from a STiC model 

    :Authors: 
        Carlos Diaz (ISP/SU 2020)
    """

    def __init__(self, datasize, dx, batch_size, logtau, stokelist, cubelist, noise):
        'Initialization'
        self.n_training_orig = datasize
        self.batch_size = batch_size
        self.dx = dx
        self.noise = noise  # CHECK THAT IM USING NOISE!
        self.logtau = logtau
        self.stokelist = np.array(stokelist)
        self.cubelist = np.array(cubelist)
        self.batchs_per_epoch_training = int(self.n_training_orig / self.batch_size)
        self.n_training = self.batchs_per_epoch_training * self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        input_train_get, output_train_get = self.__data_generation(self)
        return input_train_get, output_train_get

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batchs_per_epoch_training

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'

        stokes = self.stokelist[0, :]
        cube = self.cubelist[0, :]
        nl, ny, nx = stokes.shape
        ntau, ny, nx = cube.shape
        Lx = nx
        Ly = ny
        dx = self.dx

        # Regularization
        jitterOption = True
        mulitplyJitter = 2
        mynoisecube = 1e-2

        input_train = np.zeros((self.batch_size, dx, dx, int(nl)))
        output_train = np.zeros((self.batch_size, dx, dx, int(ntau)))
        for j in range(self.batch_size):
            randi = random.randint(0, self.stokelist.shape[0] - 1)
            stokes = self.stokelist[randi, :]

            xpos = random.randint(0, Lx - dx)
            ypos = random.randint(0, Ly - dx)
            rota = random.randint(0, 3)

            ni = len(self.logtau)
            ministokes = stokes[:,ypos:ypos + dx, xpos:xpos + dx] 

            lenq = cube.shape[0]
            minicube = np.zeros((lenq, ministokes.shape[1], ministokes.shape[2]))

            for iq in range(lenq):
                jitterX = random.randint(-1 * mulitplyJitter, +1 * mulitplyJitter)
                jitterY = random.randint(-1 * mulitplyJitter, +1 * mulitplyJitter)
                if jitterOption is False:
                    jitterY, jitterX = 0, 0

                import scipy.ndimage as nd
                minicube[iq, :, :] = nd.shift(cube[iq, ypos:ypos + dx, xpos:xpos + dx],
                                              (jitterY, jitterX), mode='nearest')

            # Extra noise
            minicube = minicube[:] + minicube * np.random.normal(0., mynoisecube,(cube.shape[0], dx, dx))
            ministokes = ministokes[:] + np.random.normal(0.,self.noise,(stokes.shape[0],dx,dx))

            from ISPy.util.azimuth import BTAZI2BQBU_cube
            minicube[ni * 4:5 * ni, :, :], minicube[ni * 5:6 * ni, :, :] = BTAZI2BQBU_cube(
                minicube[ni * 4:5 * ni, :, :], minicube[ni * 5:6 * ni, :, :])

            input_train[j, :, :, :] = gentools.rotate_cube(np.swapaxes(ministokes, 0, 2), rota)
            output_train[j, :, :, :] = gentools.rotate_cube(np.swapaxes(minicube, 0, 2), rota)

        return input_train, output_train




# ==================================================================================
def network1D(nx, ny, nd, nq, activation='relu', n_filters=64, l2_reg=1e-7):
    """ Neural network architecture 
    
    :Authors: 
        Carlos Diaz (ISP/SU 2020)

    """

    def minires(inputs, n_filters, kernel=1):
        x = Conv2D(int(n_filters), (kernel, kernel), padding='valid',
                   kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(inputs)
        x = ELU(alpha=1.0)(x)
        x = Conv2D(n_filters, (kernel, kernel), padding='valid',
                   kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(x)
        return x

    def myblock(inputs, n_filters, kernel=1):
        x = Conv2D(n_filters, (kernel, kernel), padding='valid',
                   kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(inputs)
        xo = ELU(alpha=1.0)(x)
        x = Conv2D(n_filters, (kernel, kernel), padding='valid',
                   kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(xo)
        x = ELU(alpha=1.0)(x)
        x = add([x, xo])
        return x

    inputs = Input(shape=(nx, ny, nd))  # depth goes last in TensorFlow
    nd4 = int(nd/4)

    # TEMP
    x1 = myblock(inputs, n_filters)
    x1 = minires(x1, int(nq / 6))
    # x1 = Activation('elu')(x1) 
    x1 = Lambda(lambda x: x + 5.0)(x1)

    # VLOS
    x2 = myblock(inputs, n_filters)
    x2 = minires(x2, int(nq / 6))

    # VTURB
    x3 = myblock(inputs, n_filters)
    x3 = minires(x3, int(nq / 6))
    # x3 = Activation('relu')(x3)

    # BLONG
    xV = Lambda(lambda x: concatenate([x[:,:,:, 0:nd4], x[:,:,:, 1*nd4:2*nd4], 
                                        x[:,:,:, 2*nd4:3*nd4], 100*x[:,:,:, 3*nd4:]])   )(inputs)
    x4 = myblock(xV, n_filters)
    x4 = minires(x4, int(nq / 6))

    # BHOR - BQ
    xQ = Lambda(lambda x: concatenate([x[:,:,:, 0:nd4], 100*x[:,:,:, 1*nd4:2*nd4], 
                                        x[:,:,:, 2*nd4:3*nd4], x[:,:,:, 3*nd4:]])   )(inputs)
    x5 = myblock(xQ, n_filters)
    x5 = minires(x5, int(nq / 6))

    # BHOR - BU
    xU = Lambda(lambda x: concatenate([x[:,:,:, 0:nd4], x[:,:,:, 1*nd4:2*nd4], 
                                        100*x[:,:,:, 2*nd4:3*nd4], x[:,:,:, 3*nd4:]])   )(inputs)
    x6 = myblock(xU, n_filters)
    x6 = minires(x6, int(nq / 6))

    final = concatenate([x1, x2, x3, x4, x5, x6])
    return Model(inputs=inputs, outputs=final)


# ==================================================================================
class deep_network(object):
    """Deep neural network class: it defines the network, load the weigths, does the 
    training and the predictions. 

    :Authors: 
        Carlos Diaz (ISP/SU 2020)

    """
    def __init__(self, root, logtau, nl):
        self.root = root
        self.nl = nl
        self.logtau = logtau
        self.ntau = len(self.logtau) * 6
        self.dx = 20
        self.nx, self.ny = self.dx, self.dx
        self.nworker = 16

    def define_network(self):
        print("[INFO] Setting up network from scratch")
        self.model = network1D(self.nx, self.ny, int(self.nl), int(self.ntau))

    def read_network(self):
        print("[INFO] Setting up network and loading weights {0}_weights.hdf5".format(self.root))
        self.model = network1D(self.nx, self.ny, int(self.nl), int(self.ntau))
        self.model.load_weights("{0}_weights.hdf5".format(self.root))

    def train(self, n_iterations, stokelist, cubelist, extranoise, learning_rate, batch_size,
              datasize):
        self.n_training_orig = datasize
        self.batch_size = batch_size
        self.n_validation_orig = int(batch_size)
        self.lr = learning_rate
        self.noise = extranoise
        self.batchs_per_epoch_training = int(self.n_training_orig / self.batch_size)
        self.batchs_per_epoch_validation = int(self.n_validation_orig / self.batch_size)
        self.n_training = self.batchs_per_epoch_training * self.batch_size
        self.n_validation = self.batchs_per_epoch_validation * self.batch_size

        self.model.compile(loss='mean_absolute_error', optimizer=Adam(lr=self.lr))
        print("[INFO] Training network during {} epochs:".format(n_iterations))
        losses = []
        self.checkpointer = ModelCheckpoint(filepath="{0}_weights.hdf5".format(self.root),
                                            verbose=2, save_best_only=False)

        # Generators
        training_generator_class = DataGenerator(self.n_training_orig, self.dx, self.batch_size,
                                                 self.logtau, stokelist, cubelist, self.noise)
        validation_generator_class = DataGenerator(self.n_validation_orig, self.dx, self.batch_size,
                                                   self.logtau, stokelist, cubelist, self.noise)
        
        self.metrics = self.model.fit_generator(training_generator_class,
                                                self.batchs_per_epoch_training, epochs=n_iterations,
                                                callbacks=[self.checkpointer],
                                                validation_data=validation_generator_class,
                                                validation_steps=self.batchs_per_epoch_validation,
                                                use_multiprocessing=True, workers=self.nworker)

    def read_and_predict(self, inputdata):
        print("[INFO] Setting up network for predictions")

        # print(inputdata.shape)
        self.nx = inputdata.shape[3]
        self.ny = inputdata.shape[2]
        self.nl = inputdata.shape[1]
        self.ntau = len(self.logtau) * 6

        self.model = network1D(self.nx, self.ny, int(self.nl), int(self.ntau))
        print("[INFO] Loading network weights: {0}_weights.hdf5".format(self.root))
        self.model.load_weights("{0}_weights.hdf5".format(self.root))

        input_validation = np.zeros((1, self.nx, self.ny, self.nl), dtype='float32')
        input_validation[0, :, :, :] = inputdata.T[:, :, :, 0]

        start = time.time()
        out = self.model.predict(input_validation)
        end = time.time()
        print("[INFO] Prediction took: {0:3.2} seconds".format(end - start))

        print("[INFO] Azimuth inverse transformation")
        from ISPy.util.azimuth import BQBU2BTAZI_cube
        # Inverse transformation
        out = np.reshape(out, (input_validation.shape[0], self.nx, self.ny, 6, 9))
        out[0, :, :, 4, :], out[0, :, :, 5, :] = BQBU2BTAZI_cube(out[0, :, :, 4, :], out[0, :, :, 5, :])
        out = np.reshape(out, (input_validation.shape[0], self.nx, self.ny, 54))

        return out


# ==================================================================================
class neural_estimator(object):
    """Creates a small neural network that can be trained with STiC results
    to perform faster inversions on new data. Note: data index np.where(o.weights[:, 0] < 1.0)[0]


    Example
    -------
    >>> from ISPy.util import neural_estimation as nst
    >>> import sparsetools as sp

    # Reading data in STiC format:
    >>> model_train_list = ['model.nc']
    >>> stokes_train_list = ['profiles.nc']
    >>> logtau = [-7,-6,-5,-4,-3,-2,-1, 0, 1]

    # Initializing the neural network
    >>> myestimator = nst.neural_estimator()
    >>> myestimator.train(name='network1',option='start',nepochs=40,model_train_list,stokes_train_list,logtau)
    >>> myestimator.quickplot(filename ='testplot.pdf')

    >>> dataprediction = 'newprofiles.nc'
    >>> original_logtau = sp.model(model_train_list[0],0,0,0).ltau[0,0,0,:]
    >>> myestimator.predict(name='network1',dataprediction,logtau,original_logtau,"model_output.nc")
    
    :Authors: 
        Carlos Diaz (ISP/SU 2020)

    """
    def __init__(self):
        # self.name = name
        self.init = 0
        self.num_params = 6
        self.logtau = 0
        self.nl = None

    def predict(self, name, inputdata, logtau, original_logtau, nameoutput='model_neuralnetwork.nc', pgastop = 1.0):
        """It uses a pre-trained neural network with new observed data

        Parameters
        ----------
        name : str, optional
            name of the network, by default 'network1'
        inputdata : ncfile
            input file in STiC format
        logtau : list
            logtau scale used to train the network
        original_logtau : list
            Final stratification of the model to do the interpolation
        nameoutput : str, optional
            name of the output model, by default 'model_neuralnetwork.nc'

        Example
        -------
        >>> dataprediction = 'newprofiles.nc'
        >>> original_logtau = sp.model(model_train_list[0],0,0,0).ltau[0,0,0,:]
        >>> myestimator.prediction(name='network1',dataprediction,logtau,original_logtau,"model_output.nc")
        
        """

        print('[INFO] Sending the data to the network')
        o = sp.profile(inputdata)
        idx = np.where(o.weights[:, 0] < 1.0)[0]
        stokelist = np.array([np.concatenate([o.dat[0, :, :, idx, 0], 1e0 * o.dat[0, :, :, idx, 1],
                                              1e0 * o.dat[0, :, :, idx, 2],
                                              1e0 * o.dat[0, :, :, idx, 3]])])
        print(stokelist.shape,'...')
        self.nl = stokelist.shape[1]
        self.deepl = deep_network(name, logtau, self.nl)
        prediction = self.deepl.read_and_predict(stokelist)
        nx, ny, dum = prediction[0, :, :, :].shape
        prediction = np.reshape(prediction[0, :, :, :], (nx, ny, 6, len(logtau)))
        noriginaltau = len(original_logtau)

        # Fill the model with the prediction
        print('[INFO] Writing in STiC format')
        m = sp.model(nx=nx, ny=ny, nt=1, ndep=noriginaltau)
        from tqdm import tqdm
        for ix in tqdm(range(nx)):
            for iy in range(ny):
                temp = np.interp(original_logtau, logtau, np.abs(prediction[ix, iy, 0, :]))
                vlos = np.interp(original_logtau, logtau, prediction[ix, iy, 1, :])
                vturb = np.interp(original_logtau, logtau, np.abs(prediction[ix, iy, 2, :]))
                Bln = np.interp(original_logtau, logtau, prediction[ix, iy, 3, :])
                Bho = np.interp(original_logtau, logtau, np.abs(prediction[ix, iy, 4, :]))
                Bazi = np.interp(original_logtau, logtau, prediction[ix, iy, 5, :])

                m.ltau[0, iy, ix, :] = original_logtau
                m.temp[0, iy, ix, :] = temp * 1e3
                m.vlos[0, iy, ix, :] = vlos * 1e5
                m.pgas[0, iy, ix, :] = pgastop
                m.vturb[0, iy, ix, :] = vturb * 1e5
                m.Bln[0, iy, ix, :] = Bln * 1e3
                m.Bho[0, iy, ix, :] = Bho * 1e3
                m.azi[0, iy, ix, :] = Bazi

        # Write the model
        m.write(nameoutput)

    def create_dataset(self, model_train_list, stokes_train_list, logtau_train_list):
        """Creates a dataset for training

        Parameters
        ----------
        model_train_list : list of strings
            List of models in STiC format used for training
        stokes_train_list : list of strings
            List of observed or synthetic profiles for training
        logtau_train_list : list
            List of logtau values included in the training

        """
        self.logtau = np.array(logtau_train_list)

        stokelist, cubelist = [], []
        for simu in range(len(model_train_list)):
            m = sp.model(model_train_list[simu])
            s = sp.profile(stokes_train_list[simu])
            idx = np.where(s.weights[:, 0] < 1.0)[0]
            indices = sorted(gentools.findindex(self.logtau, m.ltau[0, 0, 0, :]))
            ni = len(indices)

            # Physical parameters
            supercube = np.zeros((ni * self.num_params, m.temp.shape[1], m.temp.shape[2]))
            supercube[:ni] = m.temp[0, :, :, indices] / 1e3
            supercube[ni:2 * ni] = m.vlos[0, :, :, indices] / 1e5
            supercube[ni * 2:3 * ni] = m.vturb[0, :, :, indices] / 1e5
            supercube[ni * 3:4 * ni] = m.Bln[0, :, :, indices] / 1e3
            supercube[ni * 4:5 * ni] = m.Bho[0, :, :, indices] / 1e3
            supercube[ni * 5:6 * ni] = m.azi[0, :, :, indices]

            # Stokes parameters
            stokes = np.concatenate([s.dat[0, :, :, idx, 0], 1e0 * s.dat[0, :, :, idx, 1],
                                     1e0 * s.dat[0, :, :, idx, 2], 1e0 * s.dat[0, :, :, idx, 3]])

            stokelist.append(stokes)
            cubelist.append(supercube)

        self.cubelist = cubelist
        self.stokelist = stokelist
        self.nl = len(stokes)

        
    def prepare_training(self, name='network1', option='start', nepochs=20, extranoise=5e-4,
              learning_rate=1e-4, batch_size=100, datasize=10, samplesize=20):
        """It defines the network and start the training.

        Parameters
        ----------
        name : str, optional
            name of the network, by default 'network1'
        option : str, optional
            start or continue the previous training, by default 'start'
        nepochs : int, optional
            Number of epochs, by default 20
        extranoise : [type], optional
            Extra noise level in Stokes profiles, by default 5e-4
        learning_rate : [type], optional
            Learning rate used in Adam optimizer, by default 1e-4
        batch_size : int, optional
            Size of each batch, by default 100
        datasize : int, optional
            Size of the dataset created for training, by default 10
        samplesize : int, optional
            Size of each FOV created for training, by default 20
        
 
        """

        self.name = name
        self.option = option  # [start] or [continue]
        self.nepochs = nepochs
        self.extranoise = extranoise
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.samplesize = samplesize
        self.datasize = datasize*batch_size

        if self.logtau is None: print('Variable logtau scale should be define')
        if self.cubelist is None: print('Data to train should be given')
        if self.stokelist is None: print('Data to train should be given')

        self.deepl = deep_network(self.name, self.logtau, self.nl)

        if self.option == 'start':
            self.deepl.define_network()
            self.deepl.train(self.nepochs, self.stokelist, self.cubelist, self.extranoise,
                             self.learning_rate, self.batch_size, self.datasize)

        elif self.option == 'continue':
            self.deepl.read_network()
            self.deepl.train(self.nepochs, self.stokelist, self.cubelist, self.extranoise,
                             self.learning_rate, self.batch_size, self.datasize)

        else:
            print('Keyword "option" should be "start" or "continue"')


    def train(self, name='network1', option='start', nepochs=20, model_train_list=None, stokes_train_list=None,
            logtau_train_list=None, extranoise=6e-4, learning_rate=1e-4, batch_size=100, datasize=100, samplesize=20):
        """Train the neural network

        Parameters
        ----------
        name : str, optional
            name of the network, by default 'network1'
        option : str, optional
            start or continue the previous training, by default 'start'
        nepochs : int, optional
            Number of epochs, by default 20
        model_train_list : list of strings
            List of models in STiC format used for training
        stokes_train_list : list of strings
            List of observed or synthetic profiles for training
        logtau_train_list : list
            List of logtau values included in the training
        extranoise : [type], optional
            Extra noise level in Stokes profiles, by default 5e-4
        learning_rate : [type], optional
            Learning rate used in Adam optimizer, by default 1e-4
        batch_size : int, optional
            Size of each batch, by default 100
        datasize : int, optional
            Size of the dataset created for training, by default 10
        samplesize : int, optional
            Size of each FOV created for training, by default 20
        
        Example
        -------
        # Reading data in STiC format:
        >>> model_train_list = ['model.nc']
        >>> stokes_train_list = ['profiles.nc']
        >>> logtau = [-7,-6,-5,-4,-3,-2,-1, 0, 1]

        # Initializing the neural network
        >>> myestimator = nst.neural_estimator()
        >>> myestimator.train(name='network1',option='start',nepochs=10,model_train_list,stokes_train_list,logtau)
        >>> myestimator.quickplot()

        """


        self.create_dataset(model_train_list, stokes_train_list, logtau_train_list)
        self.prepare_training(name, option, nepochs, extranoise, learning_rate, batch_size, datasize, samplesize)



    def quickplot(self,indexlist=[3,7],filename ='testplot.pdf'):
        """Quick figure with the comparison between the training dataset and
        the prediction of the network.

        Parameters
        ----------
        indexlist : list, optional
            List of logtau to plot, by default [3,7]
        """
        print('[INFO] Running quick plot')
        import matplotlib.pyplot as plt
        from ISPy.util.plottools import phimap, add_colorbar

        stokelist = np.array(self.stokelist)[0:1,:]
        prediction = self.deepl.read_and_predict(stokelist)
        nx, ny, dum = prediction[0, :, :, :].shape

        # Prediction
        prediction = np.reshape(prediction[0, :, :, :], (nx, ny, 6, len(self.logtau)))
        cubelist =  np.array(self.cubelist).T[:,:,:,0]
        cubelist = np.reshape(cubelist, (nx,ny,6,9))
        newtau = self.logtau[:]


        listnvari = [0,2,1,3,4,5]
        cmapvari = ['magma', 'viridis', 'seismic', 'RdGy', 'PRGn', phimap]
        phylabel = ['Temp', r'$v_{turb}$',r'$v_{los}$',r'$B_{long}$',r'$B_{perp}$',r'Azi']
        limitsmax = [8.,7.,+5,+5,+10,+10,+1,+1,+1,+1,np.pi,np.pi]
        limitsmin = [4.5,5,-0,-0,-10,-10,-1,-1,-1,-1,0,0]
        normalizecte = [1.0,1.0,1.0,1.0,1.0,np.pi/180.]

        fig, axs = plt.subplots(figsize=(9,10), nrows=len(listnvari), ncols=4, sharex=True, sharey=True)
        for ii in range(len(listnvari)):
            ii2 = int(ii*2)
            ii21 = ii2 + 1

            indiplot = indexlist[0]
            axs[ii,0].set_title(r'NN - '+phylabel[ii]+r' log$\tau$={}'.format(newtau[indiplot]))
            axs[ii,0].imshow(prediction[:,:,listnvari[ii],indiplot].T*normalizecte[ii],cmap=cmapvari[ii],
            origin='lower',vmin=limitsmin[ii2],vmax=limitsmax[ii2])
            axs[ii,1].set_title(r'STiC - '+phylabel[ii]+r' log$\tau$={}'.format(newtau[indiplot]))
            im = axs[ii,1].imshow(cubelist[:,:,listnvari[ii],indiplot].T,cmap=cmapvari[ii],
            origin='lower',vmin=limitsmin[ii2],vmax=limitsmax[ii2])
            cb = add_colorbar(im, aspect=30)

            indiplot = indexlist[1]
            axs[ii,2].set_title(r'NN - '+phylabel[ii]+r' log$\tau$={}'.format(newtau[indiplot]))
            axs[ii,2].imshow(prediction[:,:,listnvari[ii],indiplot].T*normalizecte[ii],cmap=cmapvari[ii],
            origin='lower',vmin=limitsmin[ii21],vmax=limitsmax[ii21])
            axs[ii,3].set_title(r'STiC - '+phylabel[ii]+r' log$\tau$={}'.format(newtau[indiplot]))
            im = axs[ii,3].imshow(cubelist[:,:,listnvari[ii],indiplot].T,cmap=cmapvari[ii],
            origin='lower',vmin=limitsmin[ii21],vmax=limitsmax[ii21])
            cb = add_colorbar(im, aspect=30)

        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight')
