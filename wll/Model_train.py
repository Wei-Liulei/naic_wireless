import numpy as np
from tensorflow import keras
from Model_define_tf import Encoder, Decoder, NMSE
from utils import *

# plot_results,
# %% load data
data_path = '/media/D/data_set/WirelessCommunication/train'


model_save_path = './Modelsave/'
is_fit = 0
# load_model='nmse0537'
load_model='nmse059'


x_train, x_test = get_data(data_path)
x_train=x_train[:100]
# x_train = (x_train*2 -1)*10**5 -4
# x_test = (x_test*2 -1)*10**5 -4
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
# print(Encoder.summary())|
keras.utils.plot_model(autoencoder, show_shapes=True)  # , rankdir="LR"
keras.utils.plot_model(autoencoder.layers[1], show_shapes=True)  # , rankdir="LR"

# %% model load 

# load_model=False
if load_model:   
    encoder.load_weights(model_save_path + load_model + '/encoder.h5')
    decoder.load_weights(model_save_path + load_model + '/decoder.h5')
if is_fit:
    autoencoder.fit(x=x_train, y=x_train, batch_size=16, epochs=1, validation_split=0.1)
# %% model save
encoder.save(model_save_path + 'encoder.h5')
decoder.save(model_save_path + 'decoder.h5')
# autoencoder.save(model_save_path + 'autoencoder')
# %% model eval
# autoencoder = keras.models.load_model(model_save_path + 'autoencoder')
y_test = autoencoder.predict(x_test)
print('The mean NMSE is ' + np.str(NMSE(x_test, y_test)))


# plot_results(x_test, y_test, 5)
plot_results_x(x_test, 5, 10)
plot_results_v2(x_test, y_test, 5, 10)

# if __name__ == '__main__':

# %%
# import matplotlib.pyplot as plt
# i = 10
# # x_testplo_i = (x_test[i, :, :, 0]-0.5)#*10**5
# x_testplo = abs(x_test[i, :, :, 0]-0.5 + 1j*(x_test[i, :, :, 1]-0.5))
# # decoded_imgsplo = abs(x_test[i, :, :, 0]-0.5 + 1j*(x_test[i, :, :, 1]-0.5))
# plt.imshow(np.max(np.max(x_testplo))-x_testplo.T)

# x_testplo_j = (x_test[i, :, :, 1]-0.5)*10**5
# plt.imshow(np.max(np.max(x_testplo_j))-x_testplo_j.T)
