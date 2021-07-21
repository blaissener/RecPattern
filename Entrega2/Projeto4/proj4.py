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


def displace1(x, y):
    print(np.sign(x))
    print(np.sign(y))
    return [-x**2 * np.sign(x), -y**2 * np.sign(y)]

def drawFig(x, y):
    my_dpi=96
    plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)

    plt.scatter(x, y, s=1)
    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)
    #plt.savefig('my_fig.png', dpi=my_dpi)
    plt.show()

def drawQuiver(u, v):
    plt.quiver(u, v)
    plt.show()

def displacementModification(x, y, r):
        
    ###Generating circles
    circles = createCircularDistribution(x, y, r, 1000)

    x = circles[:, 0]
    y = circles[:, 1]

    ###Points after transformation by Displacement Filed
    data = displace1(x, y)

    xbar = data[0]
    ybar = data[1]
    
    ####Displacement Field
    X, Y = np.meshgrid(np.arange(-1.5, 1.5, .1), np.arange(-1.5, 1.5, .1))
    u =  -X**2 * np.sign(X)
    v =  -Y**2 * np.sign(Y)

    my_dpi=96
    plt.figure(figsize=(1000/my_dpi, 1000/my_dpi), dpi=my_dpi)
    plt.quiver(X, Y, u, v, units='width', color="blue", label="Displacement Field")
    plt.scatter(x, y, color='orange', label = "Circle", s=3)
    plt.scatter(x+xbar, y+ybar, color='gray', label = "Modified Circle", s=3)
    # plt.xlim(-1.5,1.5)
    # plt.ylim(-1.5,1.5)
    plt.legend()
    plt.show()

###Single circle radius 1 and origin centered
x = [0]
y = [0]
r = [1]

displacementModification(x, y, r)

###Two circles, radius 0.25, one centered at [-0.5, 0] and the other [0.5, 0]

# x = [-0.5, 0.5]
# y = [0, 0]
# r = [0.25, 0.25]

# circles = createCircularDistribution(x, y, r, 1000)

# x = circles[:, 0]
# y = circles[:, 1]

#drawFig(x,y)