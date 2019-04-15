import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
dic_compressed = np.load("compressed_dic_org.npy")
# dic_compressed = B_tensor_cuda.detach().cpu().numpy()
# data = np.random.randint(0, 255, size=[40, 40, 40])
plt.figure(figsize=(10,10))
# x, y, z = data[0], data[1], data[2]
ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
#  将数据点分成三部分画，在颜色上有区分度
# ax.scatter(x[:10], y[:10], z[:10], c='y')  # 绘制数据点
# ax.scatter(x[10:20], y[10:20], z[10:20], c='r')
# ax.scatter(x[30:40], y[30:40], z[30:40], c='g')
ax.scatter(dic_compressed[:,0], dic_compressed[:,1], dic_compressed[:,2], c='y')
ax.set_xlim([-1, 1])
ax.set_zlim([-1,1])
ax.set_ylim([-1, 1])

ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()

