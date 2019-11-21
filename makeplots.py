from functions import *
from sys import argv, exit

if len(argv) != 2:
    print("Usage: python {} <json file>".format(argv[0]))
    exit()

inputfile = argv[1]

fitter = Fitter(inputfile)
fitter.generateFit()

# Create a plot with two data sets together
fitter.plot(["P_Bd_im", "P_Bd_re"])

# Or loop over all form facrots in the json file
for ff in fitter.formFactors:
    fitter.plot([ff])

outputFileName = inputfile[:-5] # Strip the .json
print(fitter.meanFitParameters())
print(fitter.covarianceMatrix())
print(fitter.getResidues())

