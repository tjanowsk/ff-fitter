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

numalpha = int(argv[3])

qsqlist2 = list(range(10,26))
qsqlist = [0.00001, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 14.0, 16.0]
#names = ['Re[P*_d]','Re[Tperp*_d]','Re[Tpar*_d]', 'Im[P*_d]','Im[Tperp*_d]','Im[Tpar*_d]',\
#    'Re[P*_s]','Re[Tperp*_s]','Re[Tpar*_s]', 'Im[P*_s]','Im[Tperp*_s]','Im[Tpar*_s]']
names = ['Vperp_u', 'Vpar_u', 'Tperp_u', 'Tpar_u']
names2 = ['Vperp_d', 'Vpar_d', 'Tperp_d', 'Tpar_d']
names3 = ['Vperp_s', 'Vpar_s', 'Tperp_s', 'Tpar_s']

polelist_u = (5.325, 5.325, 5.724, 5.724)
polelist_d = (5.325, 5.325, 5.724, 5.724)
polelist_s = (5.415, 5.415, 5.829, 5.829)
filename = argv[1]

alist = list(range(3*numalpha)) + [2*numalpha] + list(range(3*numalpha, 4*numalpha - 1))

fitter = funfits(filename, names, qsqlist, numalpha=numalpha, alphalist=alist, poles = polelist_u)
lb = 1
ub = min(10, len(qsqlist))
fitter.printalphas()
fitter.genfit(lb,ub,fitform=argv[2])

fitter2 = funfits('../res_Bd_re.h5', names2, qsqlist, numalpha=numalpha, alphalist=alist, poles=polelist_d)
fitter2.genfit(lb,ub,fitform=argv[2])
fitter2.printalphas(len(fitter.alphalist))

fitter3 = funfits('../res_Bs_re.h5', names3, qsqlist, numalpha=numalpha, alphalist=alist, poles=polelist_s)
fitter3.genfit(lb,ub,fitform=argv[2])
fitter3.printalphas(len(fitter.alphalist)+len(fitter2.alphalist))

np.savetxt('results/alpha_{}_{}'.format(argv[2],argv[3]),np.concatenate((fitter.fit_cv, fitter2.fit_cv, fitter3.fit_cv)))
np.savetxt('results/cov_{}_{}'.format(argv[2],argv[3]),fitter.covalpha(fitter2,fitter3))

# Plot fits

fitter.plot('plots/{}_{}'.format(argv[2],argv[3]))
fitter2.plot('plots/{}_{}'.format(argv[2],argv[3]))
fitter3.plot('plots/{}_{}'.format(argv[2],argv[3]))

if __name__=="__main__":
    pass
