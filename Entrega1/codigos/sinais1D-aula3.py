import numpy as np
import cv2
import matplotlib.pyplot as plt

def automatoFig2(n):
	return 1.0


def visualize(automata, intMat):
	shape = intMat.shape
	nodes = shape[0]

	mat = np.zeros((100, len(automata)))

	for i in range(len(automata)):
		if(automata[i] == 0):
			mat[:,i] = 0
			
		elif(automata[i] == 1):
			mat[:,i] = 1
		
		elif(automata[i] == 2):
			mat[:,i] = 2

		else:
			mat[:,i]=3        	

	cv2.imshow("automata", mat)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def genericAutomato(n, mat): #mat must be a numpy mXm array
	automata = np.zeros(n)
	automata[0] = 0 # we must start at the node zero
	shape = mat.shape
	
	##Is it a squared matrix?
	if (shape[0] == shape[1]):
		nodes = shape[0]
	else:
		print("Not a squared matrix")
		return 1
	
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
			
	return automata

matAutomatoFig2 = np.matrix([[0.5, 0, 0, 0.7],[0.5, 0.1, 0, 0],[0, 0.9, 0.6, 0],[0, 0, 0.4, 0.3]])

data = genericAutomato(200, matAutomatoFig2)

visualize(data, matAutomatoFig2)

plt.plot(data)
plt.show()


# print(np.unique(matAutomatoFig2))