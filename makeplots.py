'''
Main file for fitting and plotting form factors
'''
from sys import argv
import numpy as np
from functions import Fitter

from components.analysis import Analysis
from components.input_reader import JsonInputReader
from components.data_reader import H5DataReader
from components.plotter import Plotter

if len(argv) != 2:
    print("Usage: python {} <json file>".format(argv[0]))
    exit()

inputfile = argv[1]

fitter = Fitter(inputfile)
fitter.generateFit()

# Create a plot with two data sets together
fitter.plot(["Tpara_Bd"])
#fitter.plot(["P_Bd_im", "P_Bd_re"])

# Or loop over all form facrots in the json file
# for ff in fitter.formFactors:
#     fitter.plot([ff])

#outputFileName = inputfile[:-5] # Strip the .json
print(fitter.meanFitParameters())
print(fitter.covarianceMatrix())
#print(fitter.getResidues())

input_file_reader = JsonInputReader(argv[1])
data_reader = H5DataReader(input_file_reader)
analysis = Analysis(input_file_reader, data_reader)
fit_results = analysis.fit()

for result in fit_results:
    print(result.full_mean())
    print(result.full_covariance())

plotter = Plotter(analysis.bare_form_factors, fit_results)
plotter.add_form_factor('Tpara_Bd', np.arange(5., 25., 0.1))
plotter.show()
