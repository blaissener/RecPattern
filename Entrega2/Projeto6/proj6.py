#using the database NAME to train a neuron web to recgnise letters

import numpy as np
import cv2
import matplotlib.pyplot as plt

#Creating figure 3

def linearDiscriminator(x, y, wx, wy, c, a):

        # shape = [mat.shape()[0],mat.shape()[0]]
        
        # for i in range(shape[0]):
        #     for j in range(shape[1]):

        #         mat[i][j] = a*(i*wx + j*wy + c)

    mat =  a*(x*wx + y*wy + c)

    return mat

x = np.arange(-1, 1, 0.01)
y = np.arange(-1, 1, 0.01)

xx, yy = np.meshgrid(x, y, sparse=True)

neuron = linearDiscriminator(xx, yy, -1, 0.5, 0.2, 1)

x = np.array([1, 2, 3, 4, 5, -6, 7, 8, 2, 5, 7])
y = np.array([5, 2, 4, -2, 1, 4, 5, 2, -1, -5, -6])
ipos = np.where(y >= 0)
ineg = np.where(y < 0)
plt.scatter(x[ipos], y[ipos], label='Positive', color='b', s=35, marker="+")
plt.scatter(x[ineg], y[ineg], label='Negative', color='r', s=35, marker="_")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Test')
plt.legend()
plt.show()