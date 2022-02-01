import numpy as np
import scipy.io as scio
from tensorflow import keras
from tensorflow.keras import metrics
from Model_define_tf import Encoder, Decoder, NMSE
import pandas as pd
# parameters
feedback_bits = 512
img_height = 126  # shape=N*126*128*2
img_width = 128
img_channels = 2

# Data loading
data_load_address = './data'
mat = scio.loadmat(data_load_address+'/Htrain.mat')
x_train = mat['H_train']
x_train = x_train.astype('float32')
mat = scio.loadmat(data_load_address+'/Htest.mat')
x_test = mat['H_test']
x_test = x_test.astype('float32')


# Model construction
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
encoder_out = encoder(autoencoder_input)
decoder_out = decoder(encoder_out)
autoencoder = keras.Model(inputs=autoencoder_input, outputs=decoder_out, name='autoencoder')
#autoencoder.compile(optimizer='adam', loss='mse')
#autoencoder.compile(optimizer='rmsprop', loss='mse')
autoencoder.compile(optimizer='rmsprop', loss="mse", metrics=[metrics.MAE, metrics.MAPE, metrics.MSE]) 
print(autoencoder.summary())
is_fit = True
#load_model = False
load_model = 'nmse0537'
if load_model:   
    encoder.load_weights(load_model + '/Modelsave/encoder.h5')
    decoder.load_weights(load_model + '/Modelsave/decoder.h5')
if is_fit:
    history = autoencoder.fit(x=x_train, y=x_train, batch_size=256, epochs=10, validation_split=0.1)
# model training
#history = autoencoder.fit(x=x_train, y=x_train, batch_size=256, epochs=1, validation_split=0.1)
    dfhistory = pd.DataFrame(history.history)
    dfhistory.to_csv("./result/history.csv")
#plot_loss(history)
# model save
# save encoder
modelsave1 = load_model + '/Modelsave/encoder.h5'
encoder.save(modelsave1)
# save decoder
modelsave2 = load_model + '/Modelsave/decoder.h5'
decoder.save(modelsave2)

# model test
y_test = autoencoder.predict(x_test)
print('The mean NMSE is ' + np.str(NMSE(x_test, y_test)))


# 以下是可视化作图部分
"""
import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display origoutal
    ax = plt.subplot(2, n, i + 1)
    x_testplo = abs(x_test[i, :, :, 0]-0.5 + 1j*(x_test[i, :, :, 1]-0.5))
    plt.imshow(np.max(np.max(x_testplo))-x_testplo.T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()
    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    decoded_imgsplo = abs(y_test[i, :, :, 0]-0.5 + 1j*(y_test[i, :, :, 1]-0.5))
    plt.imshow(np.max(np.max(decoded_imgsplo))-decoded_imgsplo.T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()
plt.show()
"""