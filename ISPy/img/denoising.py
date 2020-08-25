import numpy as np
import os
import sys
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
# To deactivate warnings: https://github.com/tensorflow/tensorflow/issues/7778
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import keras.backend.tensorflow_backend as ktf
    from keras.layers import Input, Conv2D, UpSampling2D, Concatenate, MaxPooling2D
    from keras.models import Model
    from keras.engine.topology import Layer
    from keras.engine import InputSpec
    from keras.utils import conv_utils
    import tensorflow as tf
except ImportError:
    print('Please install keras and tensorflow to continue.')
    sys.exit()

# ==================================================================================
def spatial_reflection_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
    """Pads the 2nd and 3rd dimensions of a 4D tensor.
    """
    assert len(padding) == 2
    assert len(padding[0]) == 2
    assert len(padding[1]) == 2
    if data_format is None:
        data_format = image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format ' + str(data_format))

    if data_format == 'channels_first':
        pattern = [[0, 0],
                   [0, 0],
                   list(padding[0]),
                   list(padding[1])]
    else:
        pattern = [[0, 0],
                   list(padding[0]), list(padding[1]),
                   [0, 0]]
    return tf.pad(x, pattern, "REFLECT")


# ==================================================================================
class ReflectionPadding2D(Layer):
    """Reflection-padding layer for 2D input (e.g. picture).
    This layer can add rows and columns or zeros
    at the top, bottom, left and right side of an image tensor.
    """

    def __init__(self,
                 padding=(1, 1),
                 data_format=None,
                 **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif hasattr(padding, '__len__'):
            if len(padding) != 2:
                raise ValueError('`padding` should have two elements. '
                                 'Found: ' + str(padding))
            height_padding = conv_utils.normalize_tuple(padding[0], 2,
                                                        '1st entry of padding')
            width_padding = conv_utils.normalize_tuple(padding[1], 2,
                                                       '2nd entry of padding')
            self.padding = (height_padding, width_padding)
        else:
            raise ValueError('`padding` should be either an int, '
                             'a tuple of 2 ints '
                             '(symmetric_height_pad, symmetric_width_pad), '
                             'or a tuple of 2 tuples of 2 ints '
                             '((top_pad, bottom_pad), (left_pad, right_pad)). '
                             'Found: ' + str(padding))
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            if input_shape[2] is not None:
                rows = input_shape[2] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[3] is not None:
                cols = input_shape[3] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            return (input_shape[0],
                    input_shape[1],
                    rows,
                    cols)
        elif self.data_format == 'channels_last':
            if input_shape[1] is not None:
                rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[2] is not None:
                cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            return (input_shape[0],
                    rows,
                    cols,
                    input_shape[3])

    def call(self, inputs):
        return spatial_reflection_2d_padding(inputs,
                                             padding=self.padding,
                                             data_format=self.data_format)

    def get_config(self):
        config = {'padding': self.padding,
                  'data_format': self.data_format}
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# ==================================================================================
def unet(start_ch=32, depth=2, activation='relu', input_channel_num=1, out_ch=1, inc_rate=2.,
         dropout=0.0, batchnorm=False, maxpool=True, upconv=True, residual=False):
    # UNet: code from https://github.com/pietz/unet-keras

    def _conv_block(m, dim, acti, bn, res, do=0):
        n = ReflectionPadding2D()(m)
        n = Conv2D(dim, 3, padding='valid', kernel_initializer='he_normal', activation=acti)(n)
        n = ReflectionPadding2D()(n)
        n = Conv2D(dim, 3, padding='valid', kernel_initializer='he_normal', activation=acti)(n)
        return Concatenate()([m, n]) if res else n

    def _level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
        if depth > 0:
            n = _conv_block(m, dim, acti, bn, res)
            m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
            m = _level_block(m, int(inc * dim), depth - 1, inc, acti, do, bn, mp, up, res)
            if up:
                m = UpSampling2D()(m)
                m = Conv2D(dim, 2, activation=acti, padding='same', kernel_initializer='he_normal')(
                    m)
            else:
                m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
            n = Concatenate()([n, m])
            m = _conv_block(n, dim, acti, bn, res)
        else:
            m = _conv_block(m, dim, acti, bn, res, do)
        return m

    i = Input(shape=(None, None, input_channel_num))
    o = _level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv,
                     residual)
    o = Conv2D(out_ch, 1)(o)
    return Model(inputs=i, outputs=o)


# =================================================================================
class deep_network(object):

    def __init__(self):
        self.network_type = 'network'
        self.nfilter = 32
        this_dir, this_filename = os.path.split(__file__)
        DATA_PATH = os.path.join(this_dir,
                                 "../data/{0}_denoising_weights.hdf5".format(self.network_type))
        self.model = unet(start_ch=self.nfilter)
        print("==> Setting up the network: loading {0}_weights.hdf5".format(self.network_type))
        self.model.load_weights(DATA_PATH)

    def define_network(self, image):
        self.image = image
        self.nx = image.shape[1]
        self.ny = image.shape[2]

    def predict(self, numerotime):
        input_validation = np.zeros((self.image.shape[0], self.nx, self.ny, 1), dtype='float32')
        input_validation[:, :, :, 0] = self.image
        # start = time.time()
        out = self.model.predict(input_validation)
        # end = time.time()
        # print("Prediction took {0:3.2} seconds...".format(end-start))        

        return self.image[:, :, :] - out[:, :, :, 0]


# =================================================================================
def predict_image(model, image, numerotime, split):
    # Patch for big images in keras
    if split is True:
        tamano = image.shape[1]
        index = int(tamano / 2)
        cindex = int(index / 4. + 2) * int(4)
        cindex_minor = int(index / 4. + 0) * int(4)
        ciclo = np.zeros_like(image, dtype=np.float32)

        # First part:
        print('(1/2)', end='')
        image1 = image[:, :cindex, :]
        model.define_network(image=np.nan_to_num(image1))
        ciclo1 = model.predict(numerotime)
        ciclo[:, :cindex_minor, :] = ciclo1[:, :cindex_minor, :]

        print('(2/2)', end='')
        image1 = image[:, -cindex:, :]
        model.define_network(image=np.nan_to_num(image1))
        ciclo2 = model.predict(numerotime)
        ciclo[:, -cindex_minor:, :] = ciclo2[:, -cindex_minor:, :]

    else:
        ciclo = model.predict(numerotime)
    return ciclo


# =================================================================================
def neural_network(input_data, niterations=2, scale=None, plotOption=False, test_Option=False,
                   split=False):
    """
    Run the denoising neural network algorithm developed in the paper https://arxiv.org/abs/1908.02815

    Parameters
    ----------
    input_data: 5D ndarray
        data cube
    niterations : int, optional 
        number of times the neural network, cleaning the residuals
    scale : float, optional
        normalization to match training values. The std of the normalized data has to be around 1e-3
    plotOption : bool, optional
        some plots to check the quality of the denoising.
    test_Option : bool, optional
        crop the input data to perform a quick check.
    split : bool, optional
        run the network splitting the data into multiple partitions (for big files)

    Returns
    -------
    cube_array: ndarray
        5D cube of shape [nt,ns,nw,nx,ny] after performing the denoising.
    
    Examples
    --------
    >>> from ISPy.img import denoising
    >>> from ISPy.io import solarnet

    >>> # Reading data:
    >>> data_input = solarnet.read('filename.fits')
    >>> ouput_name = 'test.fits'
    >>> output_file = denoising.neural_network(data_input, niterations = 2, scale=-1.0, plotOption = True, test_Option=True, split=True)
    >>> solarnet.write(ouput_name, output_file)

    :Authors: 
        Carlos Diaz (ISP/SU 2019)
    """

    print(input_data.shape)

    if test_Option is True:
        # To do some tests
        input_data = input_data[:1, :, :1, :, :]
        print('==> Cropping data to perform tests.')

    nt, ns, nw, nx, ny = input_data.shape
    if scale is not None and scale > 0.0:
        sc = np.copy(scale)
    else:
        # Calculating scale:
        noise_s = []
        for ii in range(input_data.shape[2]):
            noise_s.append(np.std(input_data[0, 1, ii, :, :]))
            noise_s.append(np.std(input_data[0, 2, ii, :, :]))
            noise_s.append(np.std(input_data[0, 3, ii, :, :]))
        sc = 1. / (np.min(noise_s) / 1e-3)
        print('==> Scale factor (so noise is 1e-3) = {}'.format(sc))

    new_output = np.zeros_like(input_data, dtype=np.float32)
    model = deep_network()

    if plotOption is True:
        import os
        if not os.path.exists('images'):
            os.makedirs('images')

    if test_Option is True:
        nt = 1
    else:
        nt = input_data.shape[0]

    stokes_label = ['I', 'Q', 'U', 'V']
    for istokes in [1, 2, 3]:
        # print('==> Denoising Stokes '+stokes_label[istokes])

        for jj in range(nt):
            print('==> Denoising all wavelengths of Stokes ' + stokes_label[istokes], end='')
            print(', time_frame:', jj, end='')
            print(", iter: 0", end='')
            sys.stdout.flush()

            input0 = input_data[jj, istokes, :, :int(input_data.shape[3] / 4.) * int(4),
                     :int(input_data.shape[4] / 4.) * int(4)] * sc
            numerotime = str(jj) + '_s' + stokes_label[istokes]

            model.define_network(image=np.nan_to_num(input0))
            # ciclo = out.predict(numerotime)
            ciclo = predict_image(model, input0, numerotime, split)

            for i in range(niterations):
                print(', ' + str(i + 1), end='')
                sys.stdout.flush()
                model.define_network(image=ciclo)
                # ciclo = out.predict(numerotime)
                ciclo = predict_image(model, input0, numerotime, split)
            print()

            if plotOption is True:
                medio = 3 * 2.6e-3
                lambdai = 0
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 6))
                plt.subplot(131)
                plt.title('Original - lambda:' + str(lambdai))
                plt.imshow(input0[lambdai, :, :], cmap='seismic', origin='lower',
                           interpolation='None', vmin=-medio, vmax=+medio)
                plt.subplot(132)
                plt.title('Clean image')
                plt.imshow(input0[lambdai, :, :] - ciclo[lambdai, :, :], cmap='seismic',
                           vmin=-medio, vmax=+medio, origin='lower', interpolation='None')
                plt.subplot(133)
                plt.title('Difference')
                plt.imshow(ciclo[lambdai, :, :], cmap='seismic', vmin=-medio, vmax=+medio,
                           origin='lower', interpolation='None')
                plt.savefig('images/output_t' + str(numerotime) + '_i' + str(niterations) + '.pdf',
                            bbox_inches='tight')

                if input0.shape[0] > 1.5:
                    lambdai = input0.shape[0] // 2 - 1
                    plt.figure(figsize=(12, 6))
                    plt.subplot(131)
                    plt.title('Original - lambda:' + str(lambdai))
                    plt.imshow(input0[lambdai, :, :], cmap='seismic', origin='lower',
                               interpolation='None', vmin=-medio, vmax=+medio)
                    plt.subplot(132)
                    plt.title('Clean image')
                    plt.imshow(input0[lambdai, :, :] - ciclo[lambdai, :, :], cmap='seismic',
                               vmin=-medio, vmax=+medio, origin='lower', interpolation='None')
                    plt.subplot(133)
                    plt.title('Difference')
                    plt.imshow(ciclo[lambdai, :, :], cmap='seismic', vmin=-medio, vmax=+medio,
                               origin='lower', interpolation='None')
                    plt.savefig(
                        'images/outputB_t' + str(numerotime) + '_i' + str(niterations) + '.pdf',
                        bbox_inches='tight')

            output0 = (input0[:, :, :] - ciclo[:, :, :]) / sc
            # Copying the original data
            new_output[jj, istokes, :, :, :] = input_data[jj, istokes, :, :, :]
            # Changing the new output
            new_output[jj, istokes, :, :int(input_data.shape[3] / 4.) * int(4),
            :int(input_data.shape[4] / 4.) * int(4)] = output0

    # We do not clean Stokes I
    new_output[:, 0, :, :, :] = input_data[:, 0, :, :, :]

    print('All done')

    # To avoid the TF_DeleteStatus message:
    # https://github.com/tensorflow/tensorflow/issues/3388
    ktf.clear_session()
    return new_output
