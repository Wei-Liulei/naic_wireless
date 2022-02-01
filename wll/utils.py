
import matplotlib.pyplot as plt
import numpy as np

def plot_result(x_test, y_test, n):  
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


# plot(x_test, y_test, 10)
