import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

from matplotlib.colors import Normalize
from math import sqrt
from numpy.random import randint
from numpy.random import default_rng
from pathlib import Path
from collections import Counter

def plotProduct(df, permutNames, groups):
    '''
    This function is used to plot the product of all data present in our dataset

    - args - 

    df: a pandas dataframe that has all our data
    permutNames: list with the names of each axes
    groups: pandas dataframe grouped by a certain column ("category" in our case)

    - return -

    none
    '''

    ###Creating the product (2x2) of all names to be plotted
    product = pd.DataFrame(list(itertools.product(permutNames, repeat=2)))

    ###Creating pyplot figure with 16 graphs
    fig, axs = plt.subplots(4,4, figsize=(30, 30))
    fig.subplots_adjust(hspace = 0.30, wspace=0.15)

    axs = axs.ravel()

    ###Using the product dataframe, plotting the graphs with all data
    for i in range(len(product)):
        one = "X"+str(product.loc[i][0])
        two = "X"+str(product.loc[i][1])
        for name, group in groups:
            axs[i].plot(group[one], group[two], marker="o", linestyle="", label=name)
            axs[i].set_xlabel(one)
            axs[i].set_ylabel(two)
            axs[i].legend()#loc="upper right"
    plt.show()

def split(data):
    '''
    Sample, in a random way, the data with half of it's elements. Using the sorting function
    '''
    randSample = default_rng()
    train = sorting(randSample.choice(data, int(len(data)/2), replace=False), 5)
    
    '''
    Getting all the elementes of "data" that are present in "train"
    '''
    res = (data[:, None] == train).all(-1).any(-1)
    
    '''
    Buildting the "test" array with the elements of "data" that are not(~) in the "train" array
    '''
    test = sorting(data[np.ix_(~res)], 5)

    return([train, test])

def sorting(list, value):
    
    data = np.array(list)
    
    data=data[np.argsort(data[:,value])]

    return data

def plotConfusionMatrix(confMat):
    names = ["setosa", "versicolor", "virginica"]
    guessed = ["setosa", "versicolor", "virginica"]

    fig, axs = plt.subplots(4,4, figsize=(200, 200), facecolor='white', edgecolor='k')
    fig.subplots_adjust(hspace = 1, wspace=1)
    axs = axs.ravel()

    for i in range(len(confMat)):
        
        axs[i].set_xticks(np.arange(len(true)))
        axs[i].set_yticks(np.arange(len(guessed)))
        axs[i].set_xticklabels(true)
        axs[i].set_yticklabels(guessed)

        axs[i].matshow(confMat[i][0], norm=[0, 1])#, color='blue'
        axs[i].set_xlabel("true")
        axs[i].set_ylabel("guessed")
        axs[i].set_title(str(confMat[i][1]))
        for (j, k), z in np.ndenumerate(confMat[i][0]):
            axs[i].text(k, j, '{:0.1f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    #plt.setp(axs.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    #plt.tight_layout()
    # fig.colorbar(axs)
    plt.show()

def plotMatrix(confMatInt):
    confmat = "/home/eu/AnaliseEReconhecimento/Projeto2/" + "confMats"
    Path(confmat).mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure()#figsize=(1920/96, 1080/96), dpi=96
    #ax = fig.add_subplot(111)
    names = ["setosa", "versicolor", "virginica"]

    for i in range(len(confMatInt)):

        name = str(confMatInt[i][1])
        figName = confmat + "/" + name + ".png"

        plt.xlabel("Verdadeiro")
        plt.ylabel("Classificado")
        plt.title("Parametros " + name)
        a = plt.imshow(confMatInt[i][0], vmin=0, vmax=1)
        fig.colorbar(a)
        plt.xticks(ticks=[0, 1, 2], labels=names)
        plt.yticks(ticks=[0, 1, 2], labels=names)
        for (j, k), z in np.ndenumerate(confMatInt[i][0]):
            plt.text(k, j, '{:0.1f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
        plt.savefig(figName)
        plt.clf()

def calcDistance(x0, y0, x1, y1):
    return sqrt((x0-x1)**2+(y0-y1)**2)

def distPlain(obj1, obj2, pair):
    return sqrt((obj1[pair[0]] - obj2[pair[0]])**2+(obj1[pair[1]] - obj2[pair[1]])**2)

def k_neighbors(data, indexes):
    ##using odd indexes as train and even as test
    
    newData = split(data)
    train, test = newData[0], newData[1]
    confusionMatrix = []
    
    ##For combinations of parameters
    for i in indexes:
        dist = []
        guessList = []
        ##Chose one from the test list
        for j in range(len(test)):
            
            ##calculate all the distances between the chosen one from the test list to all from the train list
            for m in range(len(train)):
                #distance = sqrt((train[j][i[0]]-test[m][i[0]])**2+(train[j][i[1]]-test[m][i[1]])**2)
                distance = calcDistance(train[j][i[0]],  train[j][i[1]], test[m][i[0]], test[m][i[1]])
                dist.append([distance, j, m, i])
            
            ##find the smallest distance
            sortedDist = sorting(dist, 0)
            
            ##true category |  guessed one
            guess = [test[j][4], train[sortedDist[0][2]][4]]
            
            if(guess[0] == guess[1]):
                guessList.append([guess, "True"])
            else:
                guessList.append([guess, "False"])
        
       
        
        mat = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        
        for q in range(len(guessList)):

            
            if(guessList[q][0][0] == "Iris-setosa"):
                
                if (guessList[q][0][1] == "Iris-setosa"):
                    mat[0,0]+=1

                if (guessList[q][0][1] == "Iris-versicolor"):
                    mat[1,0]+=1

                if (guessList[q][0][1] == "Iris-virginica"):
                    mat[2,0]+=1
                
            if(guessList[q][0][0] == "Iris-versicolor"):

                if (guessList[q][0][1] == "Iris-setosa"):
                    mat[0,1]+=1

                if (guessList[q][0][1] == "Iris-versicolor"):
                    mat[1,1]+=1

                if (guessList[q][0][1] == "Iris-virginica"):
                    mat[2,1]+=1
                
            if(guessList[q][0][0] == "Iris-virginica"):
                
                if (guessList[q][0][1] == "Iris-setosa"):
                    mat[0,2]+=1

                if (guessList[q][0][1] == "Iris-versicolor"):
                    mat[1,2]+=1

                if (guessList[q][0][1] == "Iris-virginica"):
                    mat[2,2]+=1

           
        # norm = np.linalg.norm(mat)
        mat = mat/25
        confusionMatrix.append([mat, i])
        
    
    plotMatrix(confusionMatrix)    

def knn(data, indexes):
    aux = split(data)
    train, test = aux[0], aux[1]

    #select the parameter pair that will be used in the knn classifier
    for par in indexes:
        #select an object from the test set
        for i in test:
            distBefore = 1000000
            #select an object from the train set
            for j in train:
                #calculate the distance between both objectes in the "pair" plain
                dist = distPlain(i, j, par)
                #Find the minor distance 
                if(dist < distBefore):
                    #refresh distance
                    distBefore = dist
                    #classify the object
                    classification = j[4]
            ###when all the objetcs of the train set have been searched,
            # we will have the minor distance and the guessed classification

            ###Evaluate the classification

colNames = ["X0", "X1", "X2", "X3", "Category"]
permutNames = ["0", "1", "2", "3"]
dataFrame = pd.read_csv("/home/eu/AnaliseEReconhecimento/Entrega1/Projeto2/irisDataset/iris.data", header=None, names=colNames)
groups = dataFrame.groupby("Category")
#plotProduct(dataFrame, permutNames, groups)

product = pd.DataFrame(list(itertools.product(permutNames, repeat=2)))
p = product.to_numpy()
p = p.astype(int)

numbers = np.arange(0, len(dataFrame), 1)

dataFrame["Index"] = numbers

data = dataFrame.to_numpy()

knn(data, p)
