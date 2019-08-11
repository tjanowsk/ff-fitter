from functions import *
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from sys import argv, exit
from os import makedirs
import json, re

if len(argv) != 2:
    print("Usage: python {} <hdf5 file>".format(argv[0]))
#    print("fit form is a string, either z for z-expansion or poly for polynomial fit")
#    print("num parameters is the number of fit parameters for each form factor")
    exit()

inputfile = argv[1]
#numalpha = int(argv[3])

pat = re.compile('([^/]*)\.json')
dirname = pat.search(inputfile).group(1)

with open(inputfile, "r") as f:
    input = json.load(f)

def getbounds(qsqlist, l, u):
    for i, it in enumerate(qsqlist):
        if it >= l:
            lb = i
            break
    for i, it in reversed(list(enumerate(qsqlist))):
        if it <= u:
            ub = i+1
            break
    return (lb, ub)

#numalpha = 3
#alist = list(np.insert(range(4*numalpha - 1), 3*numalpha, 2*numalpha))
d2 = { item:funfits(item, [x['name'] for x in input[item]['FF']], numalpha=input[item]['num_pars'], alphalist=eval(input[item]['alphalist'].replace('NA', "input[item]['num_pars']")), mB=input[item]['mB'], poles=[float(x['m_pole']) for x in input[item]['FF']]) for item in input }

names = list(input.keys())
#ub = input[names[0]]['ub']
#plotdir = 'plots/{}/{}_{}_q2max{}'.format(dirname,argv[2],argv[3],ub)
plotdir = 'plots/{}/'.format(dirname)
makedirs(plotdir, exist_ok=True)
shift = 0
inparam = []
for item in names:
    inparam.extend(list(d2[item].printalphas(shift)))
    d2[item].genfit(*getbounds(d2[item].qsqlist, input[item]['lb'], input[item]['ub']), fitform=input[item]['fit_form'])
    d2[item].plot(plotdir)
    shift += max(d2[item].alphalist) + 1

#resdir = 'results/{}/{}_{}_q2max_{}/'.format(dirname,argv[2],argv[3],ub)
resdir = 'results/{}/'.format(dirname)
makedirs(resdir, exist_ok=True)
with open(resdir+'params', 'w') as f:
    for x in inparam:
        f.write(' '.join(map(str, x)))
        f.write('\n')
np.savetxt(resdir+'alpha', [np.concatenate([d2[item].fit_cv for item in names])], delimiter=',', fmt='%10.5f', header='{',footer='}', comments='')
np.savetxt(resdir+'cov', d2[names[0]].covalpha(*[d2[item] for item in names[1:]]))
