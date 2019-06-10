from functions import *
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from sys import argv, exit
from os import makedirs

if len(argv) != 4:
    print("Usage: python {} <hdf5 file> <fitform> <num parameters>".format(argv[0]))
    print("fit form is a string, either z for z-expansion or poly for polynomial fit")
    print("num parameters is the number of fit parameters for each form factor")
    exit()

inputfile = argv[1]
numalpha = int(argv[3])
#qsqlist = [0.00001, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 14.0, 16.0]

class par:
    def __init__(self, lb, ub):
        self.lb = float(lb)
        self.ub = float(ub)
        self.mB = 0
        self.alphalist = ''
        self.ffnames = []
        self.pole = []

def readinput(filename):
    d = {}
    names = []
    with open(filename,'r') as f:
        for line in f:
            if line[0] == '#':
                continue
            if len(line.split()) == 5:
                name, mB, alpha_list, lb, ub = line.split()
                names.append(name)
                d[name] = par(lb,ub)
                d[name].mB = float(mB)
                d[name].alpha_list = alpha_list
            if len(line.split()) == 2:
                ff, pole = line.split()
                d[name].ffnames.append(ff)
                if pole == 'inf':
                    d[name].pole.append(np.inf)
                else:
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
alist_star = list(range(6*numalpha))
d2 = { item:funfits(item, d[item].ffnames, numalpha=numalpha, alphalist=eval(d[item].alpha_list), mB=d[item].mB, poles = d[item].pole) for item in d }

ub = d[names[0]].ub
plotdir = 'plots/{}_{}_q2max{}'.format(argv[2],argv[3],ub)
makedirs(plotdir, exist_ok=True)
shift = 0
inparam = []
for item in names:
    inparam.extend(list(d2[item].printalphas(shift)))
    d2[item].genfit(*getbounds(d2[item].qsqlist, d[item].lb, d[item].ub),fitform=argv[2])
    d2[item].plot(plotdir)
    shift += max(d2[item].alphalist) + 1

resdir = 'results/{}_{}_q2max_{}/'.format(argv[2],argv[3],ub)
makedirs(resdir,exist_ok=True)
with open(resdir+'params','w') as f:
    for x in inparam:
        f.write(' '.join(map(str,x)))
        f.write('\n')
np.savetxt(resdir+'alpha',[np.concatenate([d2[item].fit_cv for item in names])], delimiter=',',fmt='%10.5f',header='{',footer='}',comments='')
np.savetxt(resdir+'cov',d2[names[0]].covalpha(*[d2[item] for item in names[1:]]))
#np.savetxt(resdir+'cov',d2[names[0]].covalpha(*[d2[item] for item in names[1:]]), newline='},{',delimiter=',',fmt='%10.5f',header='{{',footer='}}',comments='')
