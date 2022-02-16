# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from utils import get_data


x_train, x_test = get_data('./data')

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('vertical'),
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    # tf.keras.layers.experimental.preprocessing.RandomContrast(0.5)
])

# data_augmentation = tf.keras.Sequential([
#    tf.keras.layers.experimental.preprocessing.RandomFlip('vertical'),
#    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
#    tf.keras.layers.experimental.preprocessing.RandomRotation(factor=(0.2, 0.28),seed=1)
#     # tf.keras.layers.experimental.preprocessing.RandomWidth(factor=(0.2, 0.3),seed=1),
#     # tf.keras.layers.experimental.preprocessing.RandomHeight(factor=(-0.1, 0.1),seed=1),
#     # tf.keras.layers.experimental.preprocessing.RandomContrast(factor=(0.2, 0.3),seed=1),
#     # tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=(0.2, 0.3),width_factor=(0.2, 0.3),seed=1),
#     # tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=(0.2, 0.3),width_factor=(0.2, 0.3),seed=1)
# ])


i= 1
# 原图
plt.figure()
x_testplo = abs(x_test[i, :, :, 0]-0.5 + 1j*(x_test[i, :, :, 1]-0.5))
x_testplo = np.max(np.max(x_testplo))-x_testplo.T
plt.imshow(x_testplo)


for fig in range(10):
    aug = data_augmentation(x_test).numpy()
    # print(aug.shape)
    ax = plt.subplot(2, 5, fig+1)
    x_testplo_a = abs(aug[i, :, :, 0]-0.5 + 1j*(aug[i, :, :, 1]-0.5))
    x_testplo_a = np.max(np.max(x_testplo_a))-x_testplo_a.T
    plt.imshow(x_testplo_a)