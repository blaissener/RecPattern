import patternClass as pc
import numpy as np

obj = pc.PatternRec()



###cdt22
###Implementação automatos figura 6

mat6a = np.matrix([[0.9,0.9],[0.1,0.1]]) ### Attention with the difference of notation between the stochastic matrix and the commum mat on numpy
mat6b = np.matrix([[0.2,0.2],[0.8,0.8]])
mat6c = np.matrix([[0.5,0.5],[0.5,0.5]])
mat6d = np.matrix([[0.9,0.882, 0, 0, 0, 0.01],[0.1, 0.098, 0,0,0,0], [0, 0.2, 0.2, 0.194, 0, 0], [0,0,0.8,0.776, 0,0],[0,0,0,0.03, 0.5, 0.495], [0,0,0,0,0.5, 0.495] ])
mat6e = np.matrix([[0.9,0.882, 0, 0, 0, 0.01],[0.1, 0.098, 0,0,0,0], [0, 0.2, 0.2, 0.194, 0, 0], [0,0,0.8,0.776, 0,0],[0,0,0,0.03, 0.5, 0.495], [0,0,0,0,0.5, 0.495] ])
#obj.visualizeAutomata(obj.genericAutomato(500, mat6a))
#obj.visualizeAutomata(obj.genericAutomato(500, mat6b))
#obj.visualizeAutomata(obj.genericAutomato(500, mat6c))
#obj.plotStem(obj.genericAutomato(500, mat6e, bin=1))

#obj.visualize2DAutomata(obj.genericAutomato(500, mat6a))
#obj.visualize2DAutomata(obj.genericAutomato(500, mat6b))
#obj.visualize2DAutomata(obj.genericAutomato(500, mat6a))
obj.density(mat6a, mat6b, mat6c)