import numpy as np
import sigpy.plot as pl
data = np.load("compressed_output_test.npy")
pl.LinePlot(data)
