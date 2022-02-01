
import matplotlib.pyplot as plt
import numpy as np
import time
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


# def plot_results(x_test, y_test, n):  
#     plt.figure(figsize=(20, 4))
#     for i in range(n):
#         # display origoutal
#         ax = plt.subplot(2, n, i + 1)
#         x_testplo = abs(x_test[i, :, :, 0]-0.5 + 1j*(x_test[i, :, :, 1]-0.5))
#         plt.imshow(np.max(np.max(x_testplo))-x_testplo.T)
#         # plt.gray()
#         # ax.get_xaxis().set_visible(False)
#         # ax.get_yaxis().set_visible(False)
#         # ax.invert_yaxis()
#         # display reconstruction
#         ax = plt.subplot(2, n, i + 1 + n)
#         decoded_imgsplo = abs(y_test[i, :, :, 0]-0.5 + 1j*(y_test[i, :, :, 1]-0.5))
#         plt.imshow(np.max(np.max(decoded_imgsplo))-decoded_imgsplo.T)
#         # plt.gray()
#         # ax.get_xaxis().set_visible(False)
#         # ax.get_yaxis().set_visible(False)
#         # ax.invert_yaxis()
#     plt.show()


def plot_results_v2(x_test, y_test, m, n):  
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


def plot_results_x(x_test, m, n):  
    plt.figure(figsize=(20, 4))
    for i in range(m, n):
        # display origoutal
        ax = plt.subplot(3, n-m, i + 1 - m)
        # x_testplo = abs(x_test[i, :, :, 0]-0.5 + 1j*(x_test[i, :, :, 1]-0.5))
        x_testplo_i = x_test[i, :, :, 0]-0.5
        plt.imshow(np.max(np.max(x_testplo_i))-x_testplo_i.T)
        plt.gray()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        # ax.invert_yaxis()
        # display reconstruction
        ax = plt.subplot(3, n-m, i + 1-m+n-m)
        x_testplo_j = x_test[i, :, :, 1]-0.5
        # decoded_imgsplo = abs(x_test[i, :, :, 0]-0.5 + 1j*(x_test[i, :, :, 1]-0.5))
        plt.imshow(np.max(np.max(x_testplo_j))-x_testplo_j.T)
        plt.gray()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        # ax.invert_yaxis()

        ax = plt.subplot(3, n-m, i + 1-m+2*n-2*m)
        x_testplo = abs(x_test[i, :, :, 0]-0.5 + 1j*(x_test[i, :, :, 1]-0.5))
        # decoded_imgsplo = abs(x_test[i, :, :, 0]-0.5 + 1j*(x_test[i, :, :, 1]-0.5))
        plt.imshow(np.max(np.max(x_testplo))-x_testplo.T)
        plt.gray()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        # ax.invert_yaxis()
    plt.show()

