#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 22:28:38 2022

@author: liulei 
"""
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
import scipy.io as scio

feedback_bits = 512

img_height = 126  # shape=N*126*128*2
img_width = 128
img_channels = 2

# input = layers.Input(shape=(28, 28, 1))
# 126, 128, 2
inp = layers.Input(shape=(img_height, img_width, img_channels))

# Encoder
# x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inp)
# x = layers.MaxPooling2D((2, 2), padding="same")(x)
# x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
# x = layers.MaxPooling2D((2, 2), padding="same")(x)

# v2
x = layers.Conv2D(2, 3, padding='SAME', activation='relu')(inp)
x = layers.Conv2D(2, 3, padding='SAME', activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(units=int(feedback_bits), activation='sigmoid')(x)
x = layers.Flatten()(x)

# Decoder
# x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
# x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
# x = layers.Conv2D(2, (3, 3), activation="sigmoid", padding="same")(x)

# Autoencoder
autoencoder = Model(input, x)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()


# %% load data
data_path = '/media/D/data_set/WirelessCommunication/train'

x_train = scio.loadmat(data_path+'/Htrain.mat')['H_train'].astype('float32')
x_test = scio.loadmat(data_path+'/Htest.mat')['H_test'].astype('float32')

autoencoder.compile(optimizer='adam', loss='mse')
print(autoencoder.summary())
keras.utils.plot_model(autoencoder, show_shapes=True)  # , rankdir="LR"

autoencoder.fit(x=x_train, y=x_train, batch_size=256, epochs=1, validation_split=0.1)




