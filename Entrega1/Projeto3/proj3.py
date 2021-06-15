from typing import Type
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from seaborn.palettes import color_palette
import scipy
from math import sqrt
def plotStem(automata):
		xAux = []
		for i in range(len(automata)):
			xAux.append(i)

		plt.stem(xAux, automata)
		plt.xlabel("passo")
		plt.ylabel("valor")
		plt.show()


	
def genericAutomato(n, mat, type="general", mat2=0): #mat must be a numpy mXm array
    '''
    Type can be: 
    "general"
    "binary"
    "triangular"
    
    '''

    automata = np.zeros(n)
    automata[0] = 1 # we must start at the node zero
    shape = mat.shape
    
    ##Is it a squared matrix?
    if (shape[0] == shape[1]):
        nodes = shape[0]
    else:
        print(shape[0])
        print(shape[1])
        print("Not a squared matrix")
        return 1
    

    if (type=="general"):
        ##Building the signal based on the stochastic matrix   
        for i in range(1, n): #n steps on the automato
            value = np.random.rand(1)
            current = int(automata[i-1])
            probs = mat[:, current]
            
            sumPrb = 0
            
            for j in range(nodes):
                
                sumPrb = sumPrb + probs[j]
                
                if(sumPrb >= value):
                    
                    automata[i] = j
                    
                    break

    if(type=="binary"):
        for i in range(1, n): #binary automato
            value = np.random.rand(1)
            current = int(automata[i-1])
            probs = mat[:, current]
            
            sumPrb = 0
            
            for j in range(nodes):
                
                sumPrb = sumPrb + probs[j]
                
                if(sumPrb >= value):
                    
                    if (j%2 == 0):
                        automata[i] = 0
                    else:
                        automata[i] = 1
                    
                    break	
    
    if(type=="triangular"):
        for i in range(1, n): #n steps on the automato
            value = np.random.rand(1)
            current = int(automata[i-1])
            probs = mat[:, current]
            
            sumPrb = 0
            
            for j in range(nodes):
                
                sumPrb = sumPrb + probs[j]
                
                if(sumPrb >= value):
                    
                    automata[i] = j
                    
                    break
        automata = np.where(automata==4, 2, automata)
        automata = np.where(automata==5, 1, automata)

                        
    return automata

def relativeFrequency(vector):
    v, c = np.unique(vector, return_counts=True)
    relativeF = []
    for i in range(len(v)):
        relativeF.append(c[i]/len(vector))
    
    mean = np.mean(relativeF)
    std = np.std(relativeF)

    return ([mean, std])

def derivative(vector):

    derivative = np.zeros(len(vector))
    for i in range(1, len(vector)):
        derivative[i] = vector[i] - vector[i-1]

    return(derivative)

def integration(vector):
    integration = np.zeros(len(vector))
    sum = 0
    for i in range(len(vector)):
        integration[i] = sum
        sum+=vector[i]
    return(integration[-1])



def s(vector):
    s = 0
    values, counts = np.unique(vector, return_counts=True)
    probs = np.zeros(len(values))


    for i in range(len(values)):
        probs[i] = counts[i]/len(vector)

    for i in range(len(probs)):
        s+= probs[i]*np.log2(probs[i])
    
    return([-s, 2**(-s)])
    

def intersymbleDistance(vector):
    values= np.unique(vector)
    distances = []
    idist = []
    for i in range(len(values)):
        symbol = values[i]
        for j in range(len(vector)-1):
            if(vector[j] == symbol):
                d0 = j
                for k in range(j+1, len(vector)):
                    if(vector[k] == symbol):
                        d1 = k
                        j=k
                        break
                idist.append(d1-d0)
        distances.append(np.mean(idist))
    return np.mean(distances)

def burstSize(vector):
    ##returns the mean of all symbols burst sizes
    values= np.unique(vector)
    sizes = []
    
    for i in range(len(values)):
    
        symbol = values[i]
        
        for j in range(len(vector)-1):
    
            current = vector[j]
            
            if(current == symbol):
            
                length = 0
                
                while(current == symbol and j<(len(vector)-1)):
                                        
                    length+=1
                    j+=1
                    current = vector[j]
                    
                j=length
                sizes.append(length)
    
    return np.mean(sizes)

def powerSpectrumMaxFreq(vector):
    fft = np.fft.fft(vector)
    fft = np.delete(fft, 0)[0:int(len(fft)/2)]
    conj = np.conjugate(fft)
    #freq = np.fft.fftfreq(fft.size)
    pw = fft*conj
    """ plt.stem(pw)
    plt.xlabel("Freq")
    plt.ylabel("Intencidade")
    plt.show()
    input() """
    return(np.argmax(pw))

def visibility(vector):
    
    a = np.zeros((len(vector), len(vector)))
    for j in range(2, len(vector)):
        for i in range(1,j-1):
            flag = 1
            k = i+1
            while(k<=j-1 and flag ==1):
                aux = vector[j]+(vector[i]-vector[j])*(j-k)/(j-i)
                if(vector[k]>=aux):
                    flag=0
                k = k+1
            if(flag == 1):
                a[i, j] = 1
                a[j, i] = 1
    values, ocorrence = np.unique(a, return_counts=True)

    link = ocorrence[1]/ocorrence[0]
    
    nLinks = ocorrence[1]
    
    """ plt.matshow(a, cmap="plasma")
    plt.show()
    input() """
    return(nLinks, link, cm(a))

def by2(matProbs):
    columns = ["Type", "stdRelFrq","Ent", "Even", "symbolDist", "BurstSize", "PSpecMaxFreq", "#Links", "link/possible", "Area", "CM"]
    

    #columns = ["Entropy", "Eveness", "IntersymbolDistance"]
    t = ["sawTooth", "rectangular", "triangular"]
    


    auto = []
    nameIndex = 0
    for kind in matProbs:
        
        name = t[nameIndex]
        
        if(name == "triangular"):
            ty = "triangular"
        else:
            ty = "general"

        for i in range(50):

            a = genericAutomato(200, kind, type=ty)
            entropy = s(a)
            ent = entropy[0]
            even = entropy[1]
            interS = intersymbleDistance(a)
            burst = burstSize(a)
            pwMF = powerSpectrumMaxFreq(a)
            nodes = visibility(a)
            nLinks, nodeMean, cm = nodes[0], nodes[1], nodes[2]
            std = relativeFrequency(a)[1]
            area = integration(a)
            auto.append([name, std, ent, even, interS, burst, pwMF, nLinks, nodeMean, area, cm])
            #auto.append([ent, even, interS])
        nameIndex+=1

    data = pd.DataFrame(auto, columns=columns)
    print(data)
    sb.pairplot(data, hue="Type", diag_kind="None")
    plt.show()

def cm(mat):
    cm =  scipy.ndimage.measurements.center_of_mass(mat)
    return(sqrt(cm[0]**2 + cm[1]**2))

sawTooth = np.matrix([[0.5, 0, 0, 0.7], [0.5, 0.1, 0, 0], [0, 0.9, 0.6, 0], [0, 0, 0.4, 0.3]])

triangular = np.matrix([[0.1, 0, 0, 0, 0, 0.9], [0.9, 0.2, 0, 0, 0, 0], [0, 0.8, 0.1, 0, 0, 0], [0, 0, 0.9, 0.3, 0, 0], [0, 0, 0, 0.7, 0.2, 0], [0, 0, 0, 0, 0.8, 0.1]])

rectangular = np.matrix([[0.9, 0.1], [0.1, 0.9]])

matProbs = [sawTooth, rectangular, triangular]



""" automata = genericAutomato(300, triangular, type="triangular")
#plotStem(automata)
plt.stem(automata)
plt.xlabel("Passo")
plt.ylabel("Valor")
plt.show()

 """
sawTooth2 = np.matrix([[0.1, 0, 0, 0.8], [0.9, 0.3, 0, 0], [0, 0.7, 0.6, 0], [0, 0, 0.4, 0.2]])

triangularUp = np.matrix([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
triangularDown = np.matrix([[0, 0.9, 0, 0.0], [0, 0.1, 0.9, 0], [0, 0, 0.1, 0.9], [0, 0, 0, 0.1]])




by2(matProbs)