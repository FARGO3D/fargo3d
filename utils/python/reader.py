# Displaying a binary output with python and matplotlib.

from pylab import *

filename = "gasdens2.dat"
nx = 384; ny=128

data = fromfile(filename).reshape(ny,nx)
imshow(data,origin='lower')
show()
