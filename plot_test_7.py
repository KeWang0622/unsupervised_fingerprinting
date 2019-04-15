import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sigpy.plot as pl
dic_compressed = np.load("compressed_output_test_7.npy")
# dic_compressed = B_tensor_cuda.detach().cpu().numpy()
# data = np.random.randint(0, 255, size=[40, 40, 40])
pl.LinePlot(dic_compressed)
