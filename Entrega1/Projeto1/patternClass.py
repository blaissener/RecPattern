### 
from matplotlib import markers
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path ##To create a new folder

from random import random, randrange
from random import gauss
import seaborn as sns
import cv2

class PatternRec():
	"""
		General class to be used in all pattern recognition projects
	"""
	

	### First class - Project 1, A: ####----------------------------------------------------------------------------
	def initTextReader(self, path, ext):

		self.path = path
		self.ext = ext
		self.searchDirectory(path, ext)
		self.sortFiles()

	def searchDirectory(self, path, ext):
	
		self.filesList = []

		for file in os.listdir(path):
			if file.endswith(ext):
				self.filesList.append(os.path.join(path, file))
		
	def sortFiles(self):
		number = np.empty(len(self.filesList))
		
		for i in range(len(self.filesList)):
			name = self.filesList[i].split("/")[-1].strip(self.ext)
			index = i
			number[i] = int(name.split(" ")[-1])
		
		sorted = np.argsort(number)
		filesAux = []
		for i in sorted:
			filesAux.append(self.filesList[i])
		
		self.filesList = filesAux

	def extractData(self):
		##Intended to be used with files that have 
		##"x" values on the first column and "y" values
		##on the second column.

		self.x = []
		self.y = []
				
		for file in self.filesList:
			new = open(file, "r")
			data = new.readlines()
			data.pop(0) ##data 0 is 'x'

			xAux = []
			yAux = []

			for i in range(len(data)):
				data[i].replace("\n", "")

				a, b = data[i].split(' ')
				a = a.strip('"')
				a = float(a)
				b = float(b)
				xAux.append(a)
				yAux.append(b)
			self.x.append(xAux)
			self.y.append(yAux)

	def generateBarGraphs(self):
		###Creating a new folder to put the barGraphs
		barGraphsPath = self.path + "BarGraphs"
		Path(barGraphsPath).mkdir(parents=True, exist_ok=True)

		plt.figure(figsize=(1920/96, 1080/96), dpi=96)

		for i in range(len(self.x)):
			
			name = self.filesList[i].split("/")[-1].strip(self.ext)
			figName = barGraphsPath + "/" + name + ".png"
			

			plt.xlabel("Observação")
			plt.ylabel("Medida")
			plt.title("Arquivo " + name)
			plt.bar(self.x[i], self.y[i], color='blue')
			plt.savefig(figName)
			plt.clf()

	
	def generateLineGraphs(self):
		lineGraphsPath = self.path + "LineGraphs"
		Path(lineGraphsPath).mkdir(parents=True, exist_ok=True)

		plt.figure(figsize=(1920/96, 1080/96), dpi=96)

		for i in range(len(self.x)):
			
			name = self.filesList[i].split("/")[-1].strip(self.ext)
			figName = lineGraphsPath + "/" + name + ".png"

			plt.xlabel("Observação")
			plt.ylabel("Medida")
			plt.title("Arquivo " + name)
			plt.plot(self.x[i], self.y[i], color='blue')
			plt.savefig(figName)
			plt.clf()

	def generateHistograms(self):

		histograms = self.path + "Histograms"
		Path(histograms).mkdir(parents=True, exist_ok=True)

		plt.figure(figsize=(1920/96, 1080/96), dpi=96)

		for i in range(len(self.x)):

			name = self.filesList[i].split("/")[-1].strip(self.ext)
			figName = histograms + "/" + name + ".png"

			plt.xlabel("Observação")
			plt.ylabel("Valor")
			plt.title("Arquivo " + name)
			plt.hist(self.y[i])
			plt.savefig(figName)
			plt.clf()

	def generate2DHistograms(self):

		# fig, ax = plt.subplots(tight_layout=True)
		plt.hist2d(self.x[0], self.y[0])
		plt.show()
		
	def plotBarGraphs(self):
		
		# if(self.yMean == isEmpity):
		# 	print("OKK")
		
		fig, axs = plt.subplots(5,4, figsize=(50, 50), facecolor='white', edgecolor='k')
		fig.subplots_adjust(hspace = 1, wspace=0.5)
		axs = axs.ravel()

		for i in range(len(self.x)):
			name = self.filesList[i].split("/")[-1].strip(self.ext).replace("_", "")
			axs[i].bar(self.x[i], self.y[i])#, color='blue'
			axs[i].set_xlabel("Índice Medida")
			axs[i].set_ylabel("Valor Aferido")
			axs[i].set_title(name)
		plt.show()
	
	def plotHistograms(self):

		fig, axs = plt.subplots(5,4, figsize=(50, 50), facecolor='white', edgecolor='k')
		fig.subplots_adjust(hspace = 1, wspace=0.5)
		axs = axs.ravel()

		for i in range(len(self.x)):
			name = self.filesList[i].split("/")[-1].strip(self.ext).replace("_", "")
			axs[i].hist(self.y[i])#, color='blue'
			axs[i].set_xlabel("Observação")
			axs[i].set_ylabel("Valor")
			axs[i].set_title(name)
		plt.show()


	def plotGraphs(self):
		
		fig, axs = plt.subplots(5,4, figsize=(30, 30), facecolor='g', edgecolor='k')
		fig.subplots_adjust(hspace = 0.3, wspace=0.2)
		axs = axs.ravel()

		for i in range(len(self.x)):
			name = self.filesList[i].split("/")[-1].strip(self.ext).replace("_", "")
			axs[i].plot(self.x[i], self.y[i], color='blue')
			axs[i].set_xlabel("Índice Medida")
			axs[i].set_ylabel("Valor Aferido")
			axs[i].set_title(name)
	
	def plotScatter(self):
		fig, axs = plt.subplots(5,4, figsize=(30, 30), facecolor='gray', edgecolor='k')
		fig.subplots_adjust(hspace = 0.3, wspace=0.2)
		axs = axs.ravel()

		for i in range(len(self.x)):
			name = self.filesList[i].split("/")[-1].strip(self.ext).replace("_", "")
			axs[i].scatter(self.x[i], self.y[i], color='blue')
			axs[i].set_xlabel("Índice Medida")
			axs[i].set_ylabel("Valor Aferido")
			axs[i].set_title(name)
	
	def plotScatterSorted(self):
		fig, axs = plt.subplots(5,4, figsize=(30, 30), facecolor='gray', edgecolor='k')
		fig.subplots_adjust(hspace = 0.3, wspace=0.2)
		axs = axs.ravel()

		for i in range(len(self.x)):
			name = self.filesList[i].split("/")[-1].strip(self.ext).replace("_", "")
			xAux = self.x[i]#.sort()
			yAux = sorted(self.y[i])
						
			axs[i].scatter(xAux, yAux, color='blue')
			axs[i].set_xlabel("Índice Medida")
			axs[i].set_ylabel("Valor Aferido")
			axs[i].set_title(name)
			
	def calcCov(self):
		self.covVec = []
		
		for i in range(len(self.x)):
			self.covVec.append(np.cov((self.x[i],self.y[i]))) ###Arrumar!
	
	def meanAndStd(self):
		self.yMean = [ ]
		self.yStd = [ ]
		for i in range(len(self.x)):
			self.yMean.append(np.mean(self.y[i]))
			self.yStd.append(np.std(self.y[i]))
		
		s1 = 0
		s2 = 0
		d1 = 0
		d2 = 0
		for i in range(len(self.x)):
			if (i <= 10):
				s1 += self.yMean[i]
				d1 += self.yStd[i]
			else:
				s2 += self.yMean[i]
				d2 += self.yStd[i]
		#Mean of each subset
		self.mean1 = s1/10
		self.mean2 = s2/10
		self.std1 = d1/10
		self.std2 = d2/10	

	def plotMeanStd(self):
		
		self.xGeneral = []
		for i in range(1, len(self.x)+1):
			self.xGeneral.append(i)

		
		fig, ax1 = plt.subplots(figsize=(1920/96, 1080/96), dpi=96)
		
		ax2 = ax1.twinx()
		ax1.scatter(self.xGeneral, self.yMean, color='green')
		ax2.scatter(self.xGeneral, self.yStd, color ='blue')

		ax1.set_xlabel('Arquivos')
		ax1.set_ylabel('Média', color='g')
		ax2.set_ylabel('Desvio', color='b')
		ax1.set_xticks(np.arange(min(self.xGeneral), max(self.xGeneral)+1, 1.0))
		ax1.set_yticks(np.arange(-1, 1.5, 0.5))
		ax2.set_xticks(np.arange(min(self.xGeneral), max(self.xGeneral)+1, 1.0))
		ax1.axvline(10.5, color="orange")

		plt.show()
	
	def generateUniformData(self, l=-0.5, h=0.5, r=500, n=10):
		self.uniformDistrib = []
		for i in range(n):
			self.uniformDistrib.append(np.random.uniform(low=l, high=h, size=(1,r))[0])

	def generateGaussianData(self, m=0, d=1, r=500, n=10):
		self.gaussianDistrib = []

		for i in range(n):
			mat =  np.zeros(r)
		
			for i in range(r):
				value = gauss(m, d)
				mat[i] = value
			self.gaussianDistrib.append(mat)
		print(self.gaussianDistrib)
	
	def plotGeneratedDistrib(self):
		xAux = []
		for i in range(len(self.uniformDistrib)):
			xAux.append(i)

		print(len(self.uniformDistrib))
		print(len(xAux))

		fig, axs = plt.subplots(5,4, figsize=(30, 30))
		fig.subplots_adjust(hspace = 0.3, wspace=0.2)
		axs = axs.ravel()

		for i in range(20):
			if(i<=10):
				name = "Uniforme_" + str(i)
				for j in range(len(self.uniformDistrib[0])):
					axs[i].bar(xAux[i], self.uniformDistrib[i][j], color='green')
				axs[i].set_xlabel("Observação")
				axs[i].set_ylabel("Valor")
				axs[i].set_title(name)

			else:
				name = "Gaussiana_" + str(i)
				for j in range(len(self.gaussianDistrib[0])):
					axs[i].bar(xAux[i], self.gaussianDistrib[i][j], color='green')
				axs[i].set_xlabel("Observação")
				axs[i].set_ylabel("Valor")
				axs[i].set_title(name)
		plt.show()

	def writeGeneratedData(self):
		path = "/home/eu/AnaliseEReconhecimento/Aula1/DadosGerados/"
		Path(path).mkdir(parents=True, exist_ok=True)
		for i in range(len(self.uniformDistrib)):
			file = open(path+"gen_ "+str(i)+".txt", "w")
			file.write('"'+"x"+'"'+"\n")
			for j in range(len(self.uniformDistrib[i])):
				file.write('"'+str(j)+'"' + " " + str(self.uniformDistrib[i][j])+ "\n")
			file.close()

		for i in range(len(self.gaussianDistrib)):
			file = open(path+"gen_ "+str(i + 10)+".txt", "w")
			file.write('"'+"x"+'"'+"\n")
			for j in range(len(self.gaussianDistrib[i])):
				file.write('"'+str(j)+'"' + " " + str(self.gaussianDistrib[i][j])+ "\n")
			file.close()


	def analyseGeneratedData(self):
		self.normalStats = [np.mean(self.uniformDistrib), np.std(self.uniformDistrib)]
		self.gaussianStats = [np.mean(self.gaussianDistrib), np.std(self.gaussianDistrib)]


	def generateImage(self):
		mat =  np.zeros((200, 200))
		for i in range(200):
			for j in range(200):
				value = gauss(0, 1)
				mat[i][j] = value
		cv2.imshow("randGauss", mat)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		sampl = np.random.uniform(low=-0.5, high=0.5, size=(200,200))
		cv2.imshow("uniformRand", sampl)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	####-------------------------------------------------------------------------------------------------------------
	
	### Second class - Project 1, B: ####----------------------------------------------------------------------------

	def fig5(self, n):
		automata = np.empty(n)
		automata[0] = 0
		for i in range(1, n):
			value = np.random.rand(1)
			lastValue = automata[i-1]
			if(lastValue == 0):
				if(value <= 0.1):
					automata[i] = 1 
				else:
					automata[i] = 0
			elif(lastValue == 1):
				if(value <= 0.1):
					automata[i] = 1 
				else:
					automata[i] = 0
		return automata

	def visualizeAutomata(self, automata):
		mat = np.zeros((100, len(automata)))
		zeros = 0
		ones = 0
		for i in range(len(automata)):
			if(automata[i] == 0):
				mat[:,i] = 0
				zeros += 1
			else:
				mat[:,i] = 1
				ones += 1

		cv2.imshow("automata", mat)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	
	def visualize2DAutomata(self, automata):
		mat = np.zeros((len(automata), len(automata)))
		zeros = 0
		ones = 0
		for i in range(len(automata)):
			if(automata[i] == 0):
				mat[:,i] = 0
				mat[i,:] = 0
				
			else:
				mat[:,i] = 1
				mat[i,:] = 1
				

		#cv2.imshow("automata1", mat)
		
		mat2 = np.zeros((len(automata), len(automata)))
		zeros = 0
		ones = 0
		for i in range(len(automata)):
			if(automata[i] == 0):
				# mat[:,i] = 0
				# mat[i,:] = 0
				pass
			else:
				mat2[:,i] = 1
				mat2[i,:] = 1
				

		cv2.imshow("automata2", mat2)
		print(mat2)
		
		mat3 = np.zeros((len(automata), len(automata)))
		zeros = 0
		ones = 0
		for i in range(len(automata)):
			for j in range(len(automata)-1):

				if(mat2[i][j] == 1 and mat2[i][j+1] == 1):
					cv2.circle(mat3, (i, j), 15, (255),1)
		
		
		cv2.imshow("automata3", mat3)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	
	def plotStem(self, automata):
		xAux = []
		for i in range(len(automata)):
			xAux.append(i)

		plt.stem(xAux, automata)
		plt.xlabel("passo")
		plt.ylabel("valor")
		plt.show()


	###escrever Rotina que aceita a matriz estocastica e cria o automato
	def genericAutomato(self, n, mat, bin=0): #mat must be a numpy mXm array
		automata = np.zeros(n)
		automata[0] = 0 # we must start at the node zero
		shape = mat.shape
		
		##Is it a squared matrix?
		if (shape[0] == shape[1]):
			nodes = shape[0]
		else:
			print(shape[0])
			print(shape[1])
			print("Not a squared matrix")
			return 1
		

		if (bin==0):
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

		else:
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
		return automata

	def relFreq(self, automata, target):
		c = 0
		for i in range(len(automata)):
			if (automata[i] == target):
				c+=1
		return c/len(automata)

	def plotRelFreq(self):
		
		fig = plt.gcf()
    	#fig.suptitle("f")

		
		sns.distplot(self.freqa, rug=False, hist=False, color="black")
		sns.distplot(self.freqb, rug=False, hist=False, color="yellow")
		sns.distplot(self.freqc, rug=False, hist=False, color="red")


		plt.xlim(0,1)
		plt.xlabel("F")
		plt.ylabel("Densidade")
		

	def density(self, a, b, c):

		self.freqa = []
		self.freqb = []
		self.freqc = []

		for i in (200, 500, 750, 1000, 2000):
			autoa = self.genericAutomato(i, a)
			autob = self.genericAutomato(i, b)
			autoc = self.genericAutomato(i, c)
			self.freqa.append(self.relFreq(autoa, 1))
			self.freqb.append(self.relFreq(autob, 1))
			self.freqc.append(self.relFreq(autoc, 1))

			self.plotRelFreq()
		plt.show()

	####-------------------------------------------------------------------------------------------------------------