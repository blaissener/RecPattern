import numpy as np
import os
from pathlib import Path
from vispy import plot as vp


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
			plt.bar(self.x[i], self.y[i])
			plt.savefig(figName)

	def plotWithVispy(self, x, y):

		self.fig = vp.Fig(size=(600, 500), show=False)
		line = self.fig[0, 0].plot((xa, ya), width=3, color='blue',title='Obrabo', xlabel='x',ylabel='y')

		labelgrid = self.fig[0, 0].view.add_grid(margin=10)

		colors = [(0.8, 0, 0, 1),
		          (0.8, 0, 0.8, 1),
		          (0, 0, 1.0, 1),
		          (0, 0.7, 0, 1), ]
		plot_nvals = [1, 3, 7, 31]

		box = vp.Widget(bgcolor=(1, 1, 1, 0.6), border_color='k')
		box_widget = labelgrid.add_widget(box, row=0, col=1)
		box_widget.width_max = 90
		box_widget.height_max = 120

		bottom_spacer = vp.Widget()
		labelgrid.add_widget(bottom_spacer, row=1, col=0)

		labels = [vp.Label('n=%d' % plot_nvals[i], color=colors[i], anchor_x='left')
		          for i in range(len(plot_nvals))]
		boxgrid = box.add_grid()

		grid = vp.visuals.GridLines(color=(0, 0, 0, 0.5))
		grid.set_gl_state('translucent')
		self.fig[0, 0].view.add(grid)


directory = "/home/eu/Git/RecPadroes/Exemplo1/DadosEx1/"

obj = TextReaderD(directory, ".txt")
obj.extractData()


xa = obj.x[0]
ya = obj.y[0]

obj.plotWithVispy(xa, ya)
obj.fig.show(run=True)

