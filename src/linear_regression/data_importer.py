import numpy as np

filename = 'M_data.txt'

time, magnitude = np.loadtxt(filename, unpack=True)

print(len(time))
print(len(magnitude))