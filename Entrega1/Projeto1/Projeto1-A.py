import patternClass as pc

diretory = "/home/eu/AnaliseEReconhecimento/Aula1/DadosEx1"


data = pc.PatternRec()
data.initTextReader(diretory, ".txt")
data.extractData()

data.meanAndStd()

data.generateUniformData()
data.generateGaussianData()

data.analyseGeneratedData()

#data.writeGeneratedData() #utilizado para gerar os arquivos com os dados construidos

diretory2 = "/home/eu/AnaliseEReconhecimento/Aula1/DadosGerados"

gen=pc.PatternRec()
gen.initTextReader(diretory2, ".txt")
gen.extractData()

gen.meanAndStd()

print([data.mean1, data.mean2, data.std1, data.std2])
print([gen.mean1, gen.mean2, gen.std1, gen.std2])

#data.plotScatterSorted()
#gen.plotMeanStd()
#gen.plotBarGraphs()
#gen.generateBarGraphs()
#gen.generateHistograms()
#gen.plotHistograms()
#data.generate2DHistograms()
data.generateImage()