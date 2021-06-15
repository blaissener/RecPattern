import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path ##To create a new folder
from matplotlib import colors


class TextReaderD:
	
	def __init__(self, path, ext):
		
		self.path = path
		self.ext = ext

		self.searchDirectory(path, ext)
		
	def searchDirectory(self, path, ext):
		
		self.filesList = []

		for file in os.listdir(path):
			if file.endswith(ext):
				self.filesList.append(os.path.join(path, file))


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
			plt.title("Arquivo" + self.filesList[i])
			plt.bar(self.x[i], self.y[i], color='red')
			plt.savefig(figName)

	
	def generateLineGraphs(self):
		lineGraphsPath = self.path + "LineGraphs"
		Path(lineGraphsPath).mkdir(parents=True, exist_ok=True)
		
		plt.figure(figsize=(1920/96, 1080/96), dpi=96)
		
		for i in range(len(self.x)):

			name = self.filesList[i].split("/")[-1].strip(self.ext)
			figName = lineGraphsPath + "/" + name + ".png"

			plt.xlabel("Observação")
			plt.ylabel("Medida")
			plt.title("Arquivo" + self.filesList[i])
			plt.plot(self.x[i], self.y[i], color='blue')
			plt.savefig(figName)

	def generateHistograms(self):

		histograms = self.path + "Histograms"
		Path(histograms).mkdir(parents=True, exist_ok=True)
		
		plt.figure(figsize=(1920/96, 1080/96), dpi=96)
		
		for i in range(len(self.x)):

			name = self.filesList[i].split("/")[-1].strip(self.ext)
			print(name)
			figName = histograms + "/" + name + ".png"

			plt.xlabel("Observação")
			plt.ylabel("Medida")
			plt.title("Arquivo" + self.filesList[i])
			plt.hist(self.y[i])
			plt.savefig(figName)

	def generate2DHistograms(self):

		# fig, ax = plt.subplots(tight_layout=True)
		plt.hist2d(self.x[0], self.y[0])
		plt.show()


# directory = "/home/eu/Git/RecPadroes/Exemplo1/DadosEx1/"

# obj = TextReaderD(directory, ".txt")
# obj.extractData()
# # obj.generateBarGraphs()
# # obj.generateLineGraphs()
# obj.generateHistograms()
# print(obj.y[0])