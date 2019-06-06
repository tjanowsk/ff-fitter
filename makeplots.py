from functions import *
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from sys import argv, exit

if len(argv) != 4:
    print("Usage: python {} <hdf5 file> <fitform> <num parameters>".format(argv[0]))
    print("fit form is a string, either z for z-expansion or poly for polynomial fit")
    print("num parameters is the number of fit parameters for each form factor")
    exit()

inputfile = argv[1]
numalpha = int(argv[3])
qsqlist = [0.00001, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 14.0, 16.0]

class par:
    def __init__(self, lb, ub):
        self.lb = float(lb)
        self.ub = float(ub)
        self.ffnames = []
        self.pole = []

def readinput(filename):
    d = {}
    names = []
    with open(filename,'r') as f:
        for line in f:
            if len(line.split()) == 3:
                name, lb, ub = line.split()
                names.append(name)
                d[name] = par(lb,ub)
            if len(line.split()) == 2:
                ff, pole = line.split()
                d[name].ffnames.append(ff)
                d[name].pole.append(float(pole))
    return (d, names)

d, names = readinput(inputfile)
for item in names:
    print(item, d[item].lb, d[item].ub, d[item].ffnames, d[item].pole)

def getbounds(qsqlist, l, u):
    for i, it in enumerate(qsqlist):
        if it>=l:
            lb = i
            break
    for i, it in reversed(list(enumerate(qsqlist))):
        if it<=u:
            ub = i+1
            break
    return (lb,ub)

alist = list(range(3*numalpha)) + [2*numalpha] + list(range(3*numalpha, 4*numalpha - 1))
d2 = { item:funfits(item, d[item].ffnames, qsqlist, numalpha=numalpha, alphalist=alist, poles = d[item].pole) for item in d }

shift = 0
for item in names:
    d2[item].printalphas(shift)
    d2[item].genfit(*getbounds(qsqlist, d[item].lb, d[item].ub),fitform=argv[2])
    d2[item].plot('plots/{}_{}'.format(argv[2],argv[3]))
    shift += len(d2[item].alphalist)

np.savetxt('results/alpha_{}_{}'.format(argv[2],argv[3]),np.concatenate([d2[item].fit_cv for item in names]))
np.savetxt('results/cov_{}_{}'.format(argv[2],argv[3]),d2[names[0]].covalpha(*[d2[item] for item in names[1:]]))
