import numpy as np
import scipy.io as scio
from tensorflow import keras
from Model_define_tf import Encoder, Decoder, NMSE
from utils import plot_result

# %% load data
data_path = '/media/D/data_set/WirelessCommunication/train'

x_train = scio.loadmat(data_path+'/Htrain.mat')['H_train'].astype('float32')
x_test = scio.loadmat(data_path+'/Htest.mat')['H_test'].astype('float32')


# %% build model
# parameters
feedback_bits = 512
img_height = 126  # shape=N*126*128*2
img_width = 128
img_channels = 2

Encoder_input = keras.Input(shape=(img_height, img_width, img_channels), name="encoder_input")
Encoder_output = Encoder(Encoder_input, feedback_bits)
encoder = keras.Model(inputs=Encoder_input, outputs=Encoder_output, name='encoder')

# decoder model
Decoder_input = keras.Input(shape=(feedback_bits,), name='decoder_input')
Decoder_output = Decoder(Decoder_input, feedback_bits)
decoder = keras.Model(inputs=Decoder_input, outputs=Decoder_output, name="decoder")

# autoencoder model
autoencoder_input = keras.Input(shape=(img_height, img_width, img_channels), name="original_img")
encoder_out = encoder(autoencoder_input)
decoder_out = decoder(encoder_out)
autoencoder = keras.Model(inputs=autoencoder_input, outputs=decoder_out, name='autoencoder')
autoencoder.compile(optimizer='adam', loss='mse')
print(autoencoder.summary())
keras.utils.plot_model(autoencoder, show_shapes=True)  # , rankdir="LR"

# %% model training
autoencoder.fit(x=x_train, y=x_train, batch_size=256, epochs=1, validation_split=0.1)

# %% model save
model_save_path = './Modelsave/'
encoder.save(model_save_path + 'encoder.h5')
decoder.save(model_save_path + 'decoder.h5')

# %% model test
y_test = autoencoder.predict(x_test)
print('The mean NMSE is ' + np.str(NMSE(x_test, y_test)))

plot_result(x_test, y_test, 5)
