# import matplotlib.pyplot as plt
# from pylab import cm
# # from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
# from matplotlib.ticker import LinearLocator, FormatStrFormatter


# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection="3d")
# X = np.arange(1, 10, 1)
# Y = np.arange(1, 10, 1)
# X, Y = np.meshgrid(X, Y)
# Z = 3 * X + 2 * Y + 30
# surf = ax.plot_surface(
#     X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=True
# )
# ax.set_zlim3d(0, 100)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()


# %% 
import matplotlib.pyplot as plt
import numpy as np
from pylab import cm

import scipy.io as scio
data_path = '/media/D/data_set/WirelessCommunication/train'
x_test = scio.loadmat(data_path+'/Htest.mat')['H_test'].astype('float32')

i = 9
x_testplo = x_test[i, :, :, 0]

fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(1, 129, 1)
Y = np.arange(1, 127, 1)
X, Y = np.meshgrid(X, Y)
Z = x_testplo
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=True)
ax.set_zlim3d(0, 1)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()



# %%
plt.figure()
plt.imshow(x_test[i, :, :, 0])
plt.colorbar()
plt.grid(False)
plt.show()


# import pandas as pd
# pd.DataFrame(Z).min().mean()
# plt.imshow(Z)
# plt.show()
# %%

# # -*- coding:utf-8 -*-
# # from mpl_toolkits.mplot3d import Axes3D
# # from matplotlib import cm

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X = np.arange(-5, 5, 0.1)
# Y = np.arange(-5, 5, 0.1)
# X, Y = np.meshgrid(X, Y)
# # R = np.sqrt(X ** 2 + Y ** 2)
# Z = np.sin(np.sqrt(X ** 2 + Y ** 2))


# surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# # 画表面,x,y,z坐标， 横向步长，纵向步长，颜色，线宽，是否渐变
# ax.set_zlim(-1.01, 1.01)  # 坐标系的下边界和上边界
# # ax.set_zlim(1, 100)  # 坐标系的下边界和上边界

# ax.zaxis.set_major_locator(LinearLocator(10))  # 设置Z轴标度
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))  # Z轴精度
# fig.colorbar(surf, shrink=0.5, aspect=5)  # shrink颜色条伸缩比例（0-1），aspect颜色条宽度（反比例，数值越大宽度越窄）

# plt.show()
