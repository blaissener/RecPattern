#using the database NAME to train a neuron web to recgnise letters

from matplotlib.colors import Colormap
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#Creating figure 3

def simulateFruit():
    std = 3
    number = 500
    grapes = np.zeros(number)
    bananas = np.zeros(number)

    grapeSize = abs(np.random.normal(3, std, number))
    grapeWeigh = np.random.normal(5.1, std, number)

    bananaSize = abs(np.random.normal(19, std, number))
    bananaWeigh = np.random.normal(113.3, std, number)

    data = np.zeros((2*number, 3))
    ###Grape == 0, Banana ==1
    grapeFlag = np.zeros(number)
    bananaFlag = np.ones(number)

    data[:, 0] = np.hstack((grapeSize, grapeWeigh))#[:,0]
    data[:, 1] = np.hstack((bananaSize, bananaWeigh))
    data[:number, 2] = grapeFlag
    data[number:, 2] = bananaFlag

    frame = pd.DataFrame(data = data, columns=["Size", "Weigh", "Flag"])

    print(frame)

    # plt.scatter(grapeSize, grapeWeigh, color='purple')
    # plt.scatter(bananaSize, bananaWeigh, color = "orange")

    # plt.show()
    grape  = [grapeSize, grapeWeigh]
    banana = [bananaSize, bananaWeigh]
    
    return [grape, banana]

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
    plt.fill_between(x, line, max(y), color="#3a3a3a", label = "+")
    plt.fill_between(x, min(y), line, color="gray", label = "-")
    plt.xlim([min(x), max(x)])
    plt.ylim([min(y), max(y)])
    plt.legend()
    plt.title("Descriminador Linear: " + "Wx = " + str(wx) + " " + "Wy = " + str(wy) + " " + "c = " + str(c) + " " + "A = " + str(a))
    plt.show()

def fruitDesc(x, y, wx, wy, c, a, grape, banana):

    xx, yy = np.meshgrid(x, y, sparse=True)
    neuron = linearDiscriminator(xx, yy, wx, wy, c, a)
    line = equalZero(x, wx, wy, c)

    plt.plot(x, line, linewidth=3, color="#ffc121", label="Limite")
    plt.fill_between(x, line, max(y), color="#3a3a3a", label = "Banana")
    plt.fill_between(x, min(y), line, color="gray", label = "Uva")
    plt.xlim([min(x), max(x)])
    plt.ylim([min(y), max(y)])
    plt.scatter(grape[0], grape[1], color='purple', label = "Uva")
    plt.scatter(banana[0], banana[1], color = "yellow",  label = "Banana")
    plt.xlabel("Tamanho(cm)")
    plt.ylabel("Peso(g)")
    plt.legend()
    plt.title("Descriminador Linear: " + "Wx = " + str(wx) + " " + "Wy = " + str(wy) + " " + "c = " + str(c) + " " + "A = " + str(a))
    plt.show()

x = np.round(np.arange(-1.1, 1.1, 0.1), 2)
y = np.round(np.arange(-1.1, 1.1, 0.1), 2)

fruitX = np.round(np.arange(-1, 30, 0.1), 2)
fruitY = np.round(np.arange(-10, 125, 0.1), 2)
# descriminator(x, y, -1, 0.5, 0.2, 1)
#descriminator(x, y, 1, 0.5, -0.2, 1)
# descriminator(x, y, -0.1, 0.3, 0.3, 1)
# descriminator(x, y, -0.1, -1, 1, 1)

fruits = simulateFruit()
grape = fruits[0]
banana = fruits[1]

fruitDesc(fruitX, fruitY, 2, 1, -80, 1, grape, banana)
    
# sns.heatmap(neuronDF, annot=aux, fmt = '', cbar=False)#, annot_kws=aux)
# plt.gca().invert_yaxis()

