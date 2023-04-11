import numpy as np
import matplotlib.pyplot as plt

filename = 'M_data.txt'


time, magnitude = np.loadtxt(filename, unpack=True)
print(time[0])
print(magnitude[0])

def plotting(x=time,y=magnitude, x_label='time', y_label='magnitude'):
    plt.plot(x,y,'g.')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.autoscale(enable=True, tight=True)
    plt.show()

plotting()
