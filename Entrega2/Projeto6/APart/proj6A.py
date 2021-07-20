#using the database NAME to train a neuron web to recgnise letters

from matplotlib.colors import Colormap
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#Creating figure 3


def linearDiscriminator(x, y, wx, wy, c, a):

        # shape = [mat.shape()[0],mat.shape()[0]]
        
        # for i in range(shape[0]):
        #     for j in range(shape[1]):

        #         mat[i][j] = a*(i*wx + j*wy + c)

    mat =  a*(x*wx + y*wy + c)

    return mat

def calcZero(x, wx, wy, c):
    return -(c+x*wx)/wy

def equalZero(x, wx, wy, c):
    
    y = []
    
    for i in range(len(x)):
        value = calcZero(x[i], wx, wy, c)
        y.append(value)
    
    return y

def descriminator(x, y, wx, wy, c, a):

    xx, yy = np.meshgrid(x, y, sparse=True)
    neuron = linearDiscriminator(xx, yy, wx, wy, c, a)
    line = equalZero(x, wx, wy, c)

    plt.plot(x, line, linewidth=3, color="#ffc121", label="equalZero")
    plt.fill_between(x, line, 1, color="#3a3a3a", label = "+")
    plt.fill_between(x, -1, line, color="gray", label = "-")
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.legend()
    plt.title("Descriminador Linear: " + "Wx = " + str(wx) + " " + "Wy = " + str(wy) + " " + "c = " + str(c) + " " + "A = " + str(a))
    plt.show()

x = np.round(np.arange(-1.1, 1.1, 0.1), 2)
y = np.round(np.arange(-1.1, 1.1, 0.1), 2)

descriminator(x, y, -1, 0.5, 0.2, 1)
descriminator(x, y, 1, 0.5, -0.2, 1)
descriminator(x, y, -0.1, 0.3, 0.3, 1)
descriminator(x, y, -0.1, -1, 1, 1)

# sns.heatmap(neuronDF, annot=aux, fmt = '', cbar=False)#, annot_kws=aux)
# plt.gca().invert_yaxis()

