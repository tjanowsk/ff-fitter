'''
Main file for fitting and plotting form factors
'''
from sys import argv
import numpy as np

from components.analysis import Analysis
from components.input_reader import JsonInputReader
from components.data_reader import H5DataReader
from components.plotter import Plotter

if len(argv) != 2:
    print(f'Usage:\n python {argv[0]} <json file>')
    exit()

inputfile = argv[1]

input_file_reader = JsonInputReader(argv[1])
data_reader = H5DataReader(input_file_reader)
analysis = Analysis(input_file_reader, data_reader)
fit_results = analysis.fit()

for result in fit_results:
    print(result.full_mean())
    print(result.full_covariance())

plotter = Plotter(analysis.bare_form_factors, fit_results)
plotter.add_form_factor('Tpara_Bd', np.arange(0., 14., 0.1))
plotter.show()
plotter.add_form_factor('P_Bd_re', np.arange(
    14., 25., 0.1), color='red', label='My custom P_Bd label')
plotter.add_form_factor('P_Bd_im', np.arange(14., 25., 0.1))
plotter.show()
