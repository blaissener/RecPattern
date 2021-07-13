import numpy as np
import cv2
import matplotlib.pyplot as plt


def createCircularDistribution(x, y, r, n):
    ### x, y and r must have the same length.
    ### x and y are vectors with the center coordinates of each circle
    ### and r is a vector with the radius of each circle.
    ### n is the number of points
    length = n*len(r)
    data = np.zeros((length , 2))
    counter = 0
    for i in range(len(r)):
        for _ in range(n):
            randX = np.random.uniform(-r[i], r[i]) + x[i]
            randY = np.random.uniform(-r[i], r[i]) + y[i]
            while((((randX-x[i])**2 + (randY-y[i])**2)**(0.5) > r[i])):
                randX = np.random.uniform(-r[i], r[i]) + x[i]
                randY = np.random.uniform(-r[i], r[i]) + y[i]
            data[counter][0] = randX
            data[counter][1] = randY
            counter += 1
    
    
    return(data)


x = [0, 1, 5]
y = [0, 4, 1]
r = [1, 2, 1]

circles = createCircularDistribution(x, y, r, 300)

x = circles[:, 0]
y = circles[:, 1]

print(x)
print(y)


plt.scatter(x, y)
plt.show()