
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.io as scio
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras
from Model_define_tf import Encoder, Decoder
# from Model_define_tf_lbl import Encoder, Decoder

from tensorflow.keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from pylab import cm
import scipy.io as scio


def timmer(func):
    def deco(*args, **kwargs):
        start_time = time.time()
        print(f'\n{time.strftime("%H:%M:%S", time.localtime())} {func.__name__} start running ...')
        res = func(*args, **kwargs)
        end_time = time.time()
        print(f'{time.strftime("%H:%M:%S", time.localtime())} {func.__name__} costed {(end_time-start_time):.2f} Sec')
        return res
    return deco


def get_data(data_path): 
    x_train = scio.loadmat(data_path+'/Htrain.mat')['H_train'].astype('float32')
    x_test = scio.loadmat(data_path+'/Htest.mat')['H_test'].astype('float32')
    return x_train, x_test


def plot_x(x_test, m, n):  
    # only x
    plt.figure(figsize=(20, 4))
    for i in range(m, n):
        ax = plt.subplot(3, n-m, i + 1 - m)
        x_testplo_i = x_test[i, :, :, 0]-0.5
        plt.imshow(np.max(np.max(x_testplo_i))-x_testplo_i.T)
        plt.colorbar()
        plt.gray()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        # ax.invert_yaxis()
        # display reconstruction
        ax = plt.subplot(3, n-m, i + 1-m+n-m)
        x_testplo_j = x_test[i, :, :, 1]-0.5
        # decoded_imgsplo = abs(x_test[i, :, :, 0]-0.5 + 1j*(x_test[i, :, :, 1]-0.5))
        plt.imshow(np.max(np.max(x_testplo_j))-x_testplo_j.T)
        plt.colorbar()
        plt.gray()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        # ax.invert_yaxis()
        

        ax = plt.subplot(3, n-m, i + 1-m+2*n-2*m)
        x_testplo = abs(x_test[i, :, :, 0]-0.5 + 1j*(x_test[i, :, :, 1]-0.5))
        # decoded_imgsplo = abs(x_test[i, :, :, 0]-0.5 + 1j*(x_test[i, :, :, 1]-0.5))
        plt.imshow(np.max(np.max(x_testplo))-x_testplo.T)
        plt.colorbar()
        plt.gray()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        # ax.invert_yaxis()
    plt.show()
    
# def plot_x(x_test, m, n):
import matplotlib.pyplot as plt
def plot_pq(x_test, n, p, q):  
    '''
    批量画图函数
    '''
    plt.figure(figsize=(20, 10))
    for i in range(p*q):
        ax = plt.subplot(p, q, i+1)
        x_testplo = abs(x_test[n+ i, :, :, 0]-0.5 + 1j*(x_test[n+i, :, :, 1]-0.5))
        plt.imshow(np.max(np.max(x_testplo))-x_testplo.T)
        plt.title(f'data{n+i}')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.invert_yaxis()
    plt.show()
    
# plot_pq(t.x_test, 50, 5, 10)

def plot_3dheat(x_test, i):
    x_testplo_i = x_test[i, :, :, 0]
    x_testplo_j = x_test[i, :, :, 1]
    x_testplo = abs(x_test[i, :, :, 0]-0.5 + 1j*(x_test[i, :, :, 1]-0.5))
    
    x_testplo_s = x_testplo_i - x_testplo_j
    x_testplo_a = x_testplo_i + x_testplo_j
    
    
    Z = x_testplo
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(1, 129, 1)
    Y = np.arange(1, 127, 1)
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=True)
    ax.set_zlim3d(Z.min(), Z.max())
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def plot_results(x_test, y_test, m, n):  
    # x and y 
    plt.figure(figsize=(20, 4))
    for i in range(m, n):
        # display origoutal
        ax = plt.subplot(2, n-m, i + 1 - m)
        x_testplo = abs(x_test[i, :, :, 0]-0.5 + 1j*(x_test[i, :, :, 1]-0.5))
        plt.imshow(np.max(np.max(x_testplo))-x_testplo.T)
        # plt.gray()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        # ax.invert_yaxis()
        # display reconstruction
        ax = plt.subplot(2, n-m, i + 1-m+n-m)
        decoded_imgsplo = abs(y_test[i, :, :, 0]-0.5 + 1j*(y_test[i, :, :, 1]-0.5))
        plt.imshow(np.max(np.max(decoded_imgsplo))-decoded_imgsplo.T)
        # plt.gray()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        # ax.invert_yaxis()
    plt.show()


def NMSE_np(x, x_hat):
    x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
    x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
    x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse


def nmse(x, x_hat):
    # x_real = x[:, :, :, 0].reshape(len(x), -1)
    # x_imag = x[:, :, :, 1].reshape(len(x), -1)
    # x_hat_real = x_hat[:, :, :, 0].reshape(len(x_hat), -1)
    # x_hat_imag = x_hat[:, :, :, 1].reshape(len(x_hat), -1)
       
    x_real = tf.reshape(x[:, :, :, 0], (1, -1))
    x_imag = tf.reshape(x[:, :, :, 1], (1, -1))
    x_hat_real = tf.reshape(x_hat[:, :, :, 0], (1, -1))
    x_hat_imag = tf.reshape(x_hat[:, :, :, 1], (1, -1))

    # x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
    # x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    power = K.sum(K.square(x_real-0.5) +K.square(x_imag-0.5), axis=1)
    mse = K.sum(K.square(x_real-x_hat_real) +K.square(x_imag-x_hat_imag), axis=1)
    nmse = K.mean(mse / power)
    # (1-NMSE)*100
    return (1-nmse)*100

def nmse(x, x_hat):
       
    x_real = tf.reshape(x[:, :, :, 0], (1, -1))
    x_imag = tf.reshape(x[:, :, :, 1], (1, -1))
    x_hat_real = tf.reshape(x_hat[:, :, :, 0], (1, -1))
    x_hat_imag = tf.reshape(x_hat[:, :, :, 1], (1, -1))

    power = K.sum(K.square(x_real-0.5) +K.square(x_imag-0.5), axis=1)
    mse = K.sum(K.square(x_real-x_hat_real) +K.square(x_imag-x_hat_imag), axis=1)
    nmse = K.mean(mse / power)
    return (1-nmse)*100


def mae(x, x_hat):
    return K.mean(K.abs(x_hat - x), axis=-1)


def mape(x, x_hat):
    diff = K.abs((x - x_hat) / K.maximum(K.abs(x), K.epsilon()))
    return diff

# history = pd.DataFrame(t.history.history)
# import matplotlib.pyplot as plt
def plot_his(history, metric):
    plt.figure()
    history[metric].plot(legend=True)
    history['val_'+metric].plot(legend=True)
# plot_his(history, 'mae')
# plot_his(history, 'mape')


data_augmentation = tf.keras.Sequential([
   tf.keras.layers.experimental.preprocessing.RandomFlip('vertical'),
   tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    # tf.keras.layers.experimental.preprocessing.RandomContrast(0.5)
])


def build_autoencoder():
        feedback_bits = 512
        img_height = 126  # shape=N*126*128*2
        img_width = 128
        img_channels = 2
        # encoder model
        Encoder_input = keras.Input(shape=(img_height, img_width, img_channels), name="encoder_input")
        Encoder_output = Encoder(Encoder_input, feedback_bits)
        encoder = keras.Model(inputs=Encoder_input, outputs=Encoder_output, name='encoder')
        # decoder model
        Decoder_input = keras.Input(shape=(feedback_bits,), name='decoder_input')
        Decoder_output = Decoder(Decoder_input, feedback_bits)
        decoder = keras.Model(inputs=Decoder_input, outputs=Decoder_output, name="decoder")

        # autoencoder model
        autoencoder_input = keras.Input(shape=(img_height, img_width, img_channels), name="original_img")
        
        autoencoder_input_a = data_augmentation(autoencoder_input) # wll
        encoder_out = encoder(autoencoder_input_a) # wll

        # encoder_out = encoder(autoencoder_input)  # old 
        
        decoder_out = decoder(encoder_out)
        
        autoencoder = keras.Model(inputs=autoencoder_input, outputs=decoder_out, name='autoencoder')
        return encoder, decoder, autoencoder


# %% new model
latent_dim = 512
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses


class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(32256, activation='sigmoid'),
      layers.Reshape((126, 128, 2))
    ])

  def call(self, x):
    data =  data_augmentation(x) 
    encoded = self.encoder(data)
    decoded = self.decoder(encoded)
    return decoded

def build_autoencoder2():
    autoencoder = Autoencoder(512)
    encoder, decoder = autoencoder.encoder, autoencoder.decoder,
    return encoder, decoder, autoencoder


# class Denoise(Model):
#   def __init__(self):
#     super(Denoise, self).__init__()
#     self.encoder = tf.keras.Sequential([
#       layers.Input(shape=(28, 28, 1)),
#       layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
#       layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])

#     self.decoder = tf.keras.Sequential([
#       layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
#       layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
#       layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

#   def call(self, x):
#     encoded = self.encoder(x)
#     decoded = self.decoder(encoded)
#     return decoded

# autoencoder = Denoise()
