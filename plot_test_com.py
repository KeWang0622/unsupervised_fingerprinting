import numpy as np
import sigpy.plot as pl
from optparse import OptionParser
parser = OptionParser()
parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
(options, args) = parser.parse_args()
data = np.load(options.load)
pl.LinePlot(data)
