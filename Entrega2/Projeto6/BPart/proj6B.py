import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
import platform
import numpy as np
import cv2
from numpy.core.fromnumeric import nonzero
import pandas as pd
import seaborn as sns



def loadSet(names):
    images, labels = loadlocal_mnist(
                images_path=names[0], 
                labels_path=names[1])

    images = images.reshape(60000, 28, 28)

    imagesTest = images[int(images.shape[0]/2): -1]
    labelsTest = labels[int(labels.shape[0]/2): -1]

    imagesTrain = images[0:int(images.shape[0]/2)]
    labelsTrain = labels[0:int(labels.shape[0]/2)]

    digitsTrain, numberOfEachDigitTrain = np.unique(labelsTrain, return_counts=1)
    digitsTest, numberOfEachDigitTest = np.unique(labelsTest, return_counts=1)
    
    digitsSortedTrain = np.argsort(labelsTrain)
    digitsSortedTest = np.argsort(labelsTest)

    sortedTrain = imagesTrain[digitsSortedTrain]
    sortedTest = imagesTest[digitsSortedTest]


    return [sortedTrain, labelsTrain[digitsSortedTrain], sortedTest, labelsTest[digitsSortedTest],  digitsTrain, numberOfEachDigitTrain, digitsTest, numberOfEachDigitTest ]


def meanSet(digits, numberOfEachDigit ,sortedTrain):
    general = []

    last = 0

    lastIndex = 0

    for i in range(len(digits)):
        mean = np.zeros(sortedTrain[0].shape)
        lastIndex = last
        # print("lastIndex " + str(lastIndex))
        for j in range(lastIndex, lastIndex+numberOfEachDigit[i]):
            mean += sortedTrain[j]
            last = j
        general.append(mean/numberOfEachDigit[i])
        if(i==0):
            erosion_size = 1
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (2 * erosion_size + 1, 2 * erosion_size + 1),(erosion_size, erosion_size))
            general[i] = cv2.erode(general[i], element)
        # plt.imshow(general[-1], cmap="gist_gray")
        # plt.show()
    print("Average calculated with " + str(len(sortedTrain)) + " images")
    return general


###Creating weights
def createWeights(general):
    weights = []
    for i in range(len(general)):
        weights.append(np.hstack(general[i]))

    return weights

def createStimulus(images):
    stim = []
    for i in range(len(images)):
        stim.append(np.hstack(images[i]))

    return stim

def norm(vector):
    norm = []
    for i in range(len(vector)):
        norm.append(np.linalg.norm(vector[i]))
    return norm

def projection(weights, stim, labelsTest):

    guess = []

    for i in range(len(stim)):
        proj = []
        for j in range(len(weights)):
            proj.append(np.dot(stim[i], weights[j]))
        #print(proj)
        ansatz = np.argmax(proj)
        guess.append([ansatz, labelsTest[i]])
    #print(guess)
    return guess

def createConfusionMat(guessed,numberOfEachDigitTest):
    a = np.zeros((10,10))
    for i in range(len(guessed)):

        a[guessed[i][0], guessed[i][1]] += 1
    # a = np.round(a, decimals=1)
    # print(numberOfEachDigitTest[8])
    # for i in range(len(a)):
    #     a[i:] = a[i:]/numberOfEachDigitTest


    # aMax = a.max()
    # aMin = a.min()

    # aNorm = (a - aMin ) / (aMax - aMin)
    
    df_cm = pd.DataFrame(a, index = [i for i in "0123456789"],columns = [i for i in "0123456789"])
    #plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True, cmap="gist_gray",  fmt='g', linewidths=.5, cbar=False)
    plt.xlabel("Real") 
    plt.ylabel("Inferido")
    plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
    plt.title("noErodeZero")
    plt.show()
    '''
    To normalize:
    zi=xi−min(x)/max(x)−min(x)
    '''

def chooseOne(sortedTrain, numberOfEachDigitTest):

    choosen = []
    last = 0
    
    for i in range(len(numberOfEachDigitTrain)):
        index = 0
        
        if(i == 0):
            choosen.append(sortedTrain[i])
        
        else:
            for j in range(0, i):
                index += numberOfEachDigitTrain[j]
            choosen.append(sortedTrain[index])
        
        # plt.imshow(choosen[-1])
        # plt.show()

    return choosen

def sigmoidDiscrim(weights, stim, labelsTest):

    guess = []
    
    

    for i in range(len(stim)):
        proj = []
        s = []
        out = []
        for j in range(len(weights)):
            s.append(np.dot(stim[i], weights[j]))
        
        a = max(s)
        b = min(s)

        aNorm = (s - b ) / (a - b)
        
        for i in range(len(aNorm)):
            out.append(sigmoidFunc(aNorm[i], 5))
        
        if(i == 2963):
            print(aNorm)
            print(out)
            plt.plot(aNorm, out)
            plt.xlabel("In(S)")
            plt.ylabel("Out(A(S))")
            plt.show()
            #scatter.append([dotProd, thresh])
        #     if (dotProd>= thresh):
        #         proj.append(dotProd)
        #     else:
        #         proj.append(0)
        # #print(proj)
        
        #ansatz = np.argmax(proj)
        #guess.append([ansatz, labelsTest[i]])
    #print(scatter)
    
    return guess

def sigmoidFunc(dot, b):
    
    return (1/(1+np.exp(-b*dot)) - 0.5)

def sigmoid(x, b):
    #a = 1/(1+np.exp(-b*x))
    
    for i in range(0, 20, 3):
        a = (1/(1+np.exp(-i*x)) - 0.5) #* 1/110
        plt.plot(x, a, label=r'$\beta = $' + str(i))
    #plt.xlim([-5, 5])
    #plt.ylim([-5, 5])
    plt.legend()
    plt.xlabel("In (S)")
    plt.ylabel("Out")
    plt.title("Sigmoid "r'$P = A_{(s)} = \frac{1}{1+\exp^{-s\beta}} - 0.5$')
    plt.grid()
    plt.show()

# x = np.round(np.arange(-1.1, 1.1, 0.01), 2)

# sigmoid(x, 15.5)

####Number of digits-1 is de index of the last of each digit image
# plt.imshow(imagesSorted[5922])#x[digitsSorted[i]]
# plt.show()
# plt.imshow(imagesSorted[5923])
# plt.show()


names = ["train-images.idx3-ubyte", "train-labels.idx1-ubyte"]

sortedTrain, labelsTrain, sortedTest, labelsTest, digitsTrain, numberOfEachDigitTrain, digitsTest, numberOfEachDigitTest  = loadSet(names)

print(numberOfEachDigitTest)

average = meanSet(digitsTrain, numberOfEachDigitTrain , sortedTrain)

stimulus = createStimulus(sortedTest)

choosen = chooseOne(sortedTrain,numberOfEachDigitTrain)

weights = createWeights(average)

#guessed = projection(weights, stimulus, labelsTest)

guessedSig = sigmoidDiscrim(weights, stimulus, labelsTest)

#createConfusionMat(guessed, numberOfEachDigitTest)

createConfusionMat(guessedSig, numberOfEachDigitTest)



###Confusion Matrix
# array = [[33,2,0,0,0,0,0,0,0,1,3], 
#         [3,31,0,0,0,0,0,0,0,0,0], 
#         [0,4,41,0,0,0,0,0,0,0,1], 
#         [0,1,0,30,0,6,0,0,0,0,1], 
#         [0,0,0,0,38,10,0,0,0,0,0], 
#         [0,0,0,3,1,39,0,0,0,0,4], 
#         [0,2,2,0,4,1,31,0,0,0,2],
#         [0,1,0,0,0,0,0,36,0,2,0], 
#         [0,0,0,0,0,0,1,5,37,5,1], 
#         [3,0,0,0,0,0,0,0,0,39,0], 
#         [0,0,0,0,0,0,0,0,0,0,38]]
# df_cm = pd.DataFrame(array, index = [i for i in "ABCDEFGHIJK"],
#                   columns = [i for i in "ABCDEFGHIJK"])
# plt.figure(figsize = (10,7))
# sns.heatmap(df_cm, annot=True)


'''

###checkNorm
# avgNorm = norm(average)

# choosenNorm = norm(choosen)

# weightsNorm = norm(weights)

# print("avgNorm")
# print(avgNorm)

# print("weightsNorm")
# print(weightsNorm)

# print("choosenNorm")
# print(choosenNorm)

'''

'''
###Visualization
for i in range(len(average)):
    plt.imshow(average[i])
    plt.show()

for i in range(len(weights)):
    cv2.imshow("str(i)", weights[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''

# erosion_size = 1
# element = cv2.getStructuringElement(cv2.MORPH_CROSS, (2 * erosion_size + 1, 2 * erosion_size + 1),(erosion_size, erosion_size))


#imgTest = np.hstack(imagesSorted[])

#print(np.linalg.norm(imgTest))

# plt.imshow(imagesSorted[])
# plt.show()
# cv2.imshow("str(i)", imgTest)
# cv2.imshow("str(i2)", weights[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# values = []
# nonZ = []


# for i in range(len(digits)):
    
#     print(np.argmax(np.linalg.norm(np.dot(weights[i], imgTest))))
#     nonZ.append(np.count_nonzero(weights[i] - imgTest))
    # values.append(abs(np.sum(weights[i] - imgTest)))

#
# print("----------------")
# print(np.argmax(values))
# print(np.argmin(values))

# x = np.matrix([[1, 2], [4, 3]])
# print(x.sum(axis = 0))

# lineW = []
# columnW = []


# for i in range(len(general)):
#     columnWheits = general[i].sum(axis=1)
#     columnW.append(columnWheits)
#     lineWheits = general[i].sum(axis=0)
#     lineW.append(lineWheits)
    
# print(lineW[0])
    
    # print(general[i])
    # cv2.imshow(str(i), general[i])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # plt.imshow(columnW[i], cmap="gray")
    # plt.show()
    # plt.imshow(lineW[i], cmap="gray")
    # plt.show()
    ##erode
    # erosion_size = 1
    # element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erosion_size + 1, 2 * erosion_size + 1),(erosion_size, erosion_size))
    #erosion_dst = cv2.erode(general[i], element)
    ##threshold
    # ret,thresh1 = cv2.threshold(general[i],80,255,cv2.THRESH_BINARY)
    ##histEquilize
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # cl = clahe.apply(image)

    # plt.imshow(cl, cmap="gray")
    # plt.show()