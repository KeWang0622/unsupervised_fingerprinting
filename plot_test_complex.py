import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io
dic_compressed = np.load("compressed_output_test_complex.npy")
# dic_compressed = np.load("compressed_dic_final.npy")
#dic_compressed = B_tensor_cuda.detach().cpu().numpy()
mrf_dict = scipy.io.loadmat('/mikRAID/jtamir/projects/MRF_direct_contrast_synthesis/data/DictionaryAndSequenceInfo/fp_dictionary.mat')
# print(MRF_dic.keys())
fp_dict = mrf_dict['fp_dict']
t1_list = mrf_dict['t1_list']
t2_list = mrf_dict['t2_list']
# data = np.random.randint(0, 255, size=[40, 40, 40])
fig = plt.figure(figsize=(13,10))
# x, y, z = data[0], data[1], data[2]
ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
#  将数据点分成三部分画，在颜色上有区分度
# ax.scatter(x[:10], y[:10], z[:10], c='y')  # 绘制数据点
# ax.scatter(x[10:20], y[10:20], z[10:20], c='r')
# ax.scatter(x[30:40], y[30:40], z[30:40], c='g')
p = ax.scatter(dic_compressed[:,0], dic_compressed[:,1], dic_compressed[:,2], c=t1_list[:,0],marker = '*',linewidths=0.3,cmap = "hot")
ax.set_xlim([-1, 1])
ax.set_zlim([-1,1])
ax.set_ylim([-1, 1])
fig.colorbar(p)
ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')


fig1 = plt.figure(figsize=(13,10))
# x, y, z = data[0], data[1], data[2]
ax1 = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
#  将数据点分成三部分画，在颜色上有区分度
# ax.scatter(x[:10], y[:10], z[:10], c='y')  # 绘制数据点
# ax.scatter(x[10:20], y[10:20], z[10:20], c='r')
# ax.scatter(x[30:40], y[30:40], z[30:40], c='g')
p1 = ax1.scatter(dic_compressed[:,0], dic_compressed[:,1], dic_compressed[:,2], c=t2_list[:,0],marker = '*',linewidths=0.3,cmap = "hot")
ax1.set_xlim([-1, 1])
ax1.set_zlim([-1,1])
ax1.set_ylim([-1, 1])
fig1.colorbar(p1)
ax1.set_zlabel('Z')  # 坐标轴
ax1.set_ylabel('Y')
ax1.set_xlabel('X')
plt.show()

