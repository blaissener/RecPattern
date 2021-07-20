import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
import platform
import numpy as np
import cv2
from numpy.core.fromnumeric import nonzero

xTrain, yTrain = loadlocal_mnist(
            images_path='train-images.idx3-ubyte', 
            labels_path='train-labels.idx1-ubyte')

print(xTrain.shape)

x = xTrain.reshape(60000, 28, 28)


xTest = xTrain[int(xTrain.shape[0]/2),-1]
xTrain = xTrain[0, int(xTrain.shape[0]/2)]

print(xTest.shape)
print(xTrain.shape)


digits, numberOfEachDigit = np.unique(yTrain, return_counts=1)

# print(digits)
print(numberOfEachDigit)

digitsSorted = np.argsort(yTrain)

imagesSorted = x[digitsSorted]

####Number of digits-1 is de index of the last of each digit image
# plt.imshow(imagesSorted[5922])#x[digitsSorted[i]]
# plt.show()
# plt.imshow(imagesSorted[5923])
# plt.show()

general = []

last = 0

lastIndex = 0

erosion_size = 1
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (2 * erosion_size + 1, 2 * erosion_size + 1),(erosion_size, erosion_size))

for i in range(len(digits)):
    mean = np.zeros(imagesSorted[0].shape)
    lastIndex = last
    # print("lastIndex " + str(lastIndex))
    for j in range(lastIndex, lastIndex+numberOfEachDigit[i]):
        mean += imagesSorted[j]
        last = j
    general.append(mean/numberOfEachDigit[i])
    print(np.count_nonzero(general[i]))


# for i in range(len(digits)):
#     plt.imshow(general[i])
#     plt.show()

###Creating weights
weights = []
for i in range(len(digits)):
    weights.append(np.hstack(general[i]))



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