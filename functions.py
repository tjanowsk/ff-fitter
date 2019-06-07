import numpy as np
import h5py
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

class funfits:
    def __init__(self, filename, names, qsqlist, numalpha=3, alphalist=None,  mB=5.28, poles=None, mV=0.77, pole=True):
        self.names = names
        self.fflist = {x:0 for x in names}
        with h5py.File(filename,'r') as f:
            self.data = f['results'][...]
            for n,m in zip(names, np.array_split(self.data, len(names), axis=1)):
                self.fflist[n] = m
        self.qsqlist = qsqlist
        self.mB = mB
        #self.mBstar = mBstar
        if poles == None:
            self.poles = np.array(len(names)*[np.inf])
        else:
            self.poles = poles
        self.mV = mV
        self.numalpha = numalpha
        self.pole = pole
        if alphalist == None:
            self.alphalist = list(range(len(names)*numalpha))
        else:
            self.alphalist = alphalist

    def printalphas(self, shift=0):
        return((ff,p,[x+shift for x in self.alphalist[self.numalpha*i:self.numalpha*(i+1)]]) for i,(p,ff) in enumerate(zip(self.poles, self.fflist)))
        for i, (p, ff) in enumerate(zip(self.poles,self.fflist)):
            alphas = self.alphalist[self.numalpha*i:self.numalpha*(i+1)] 
            print(ff,p,[x+shift for x in alphas])


    def z(self,qsq):
        tplus = (self.mB+self.mV)**2
        tminus =(self.mB-self.mV)**2
        t0 = tplus*(1-np.sqrt(1-tminus/tplus))
        return (np.sqrt(tplus-qsq)-np.sqrt(tplus-t0))/(np.sqrt(tplus-qsq) + np.sqrt(tplus-t0))

    def fitfun(self, qsq, *alpha):
        xargs = self.z(qsq)-self.z(0)
        if(self.pole):
            return self.piecewise(xargs, *alpha)/(1-qsq/self.mBstar**2)
        else:
            return self.piecewise(xargs, *alpha)

    def fitfun2(self, qsq, *alpha):
        xargs = self.z(qsq)-self.z(0)
        if(self.pole):
            return self.poly(xargs, *alpha)/(1-qsq/self.mBstar**2)
        else:
            return self.poly(xargs, *alpha)

    def poly(self, z, *alpha):
        ''' Returns a polynomial \alpha_i z^i'''
        res = 0
        for i, an in enumerate(alpha):
            res += an*z**i
        return res

    def piecewise(self, z, *alpha):
        res = []
        for i in range(len(self.fflist)):
            numq = self.ub - self.lb #len(self.qsqlist)
            alphas = [alpha[x] for x in self.alphalist[self.numalpha*i:self.numalpha*(i+1)] ]
            res.extend(self.poly(z[numq*i:numq*(i+1)], *alphas))
        return np.array(res)

    def genfit(self, lb, ub, fitform='z'):
        if fitform=='z':
            fun = self.fitfun
            self.fitted = self.fitfun2
        else:
            fun = self.piecewise
            self.fitted = self.poly
        self.lb = lb
        ub = min(ub,len(self.qsqlist))
        self.ub = ub
        self.mBstar = np.concatenate([(ub-lb)*[x] for x in self.poles])
        data = np.concatenate([self.fflist[x][:,lb:ub] for x in self.names], axis=1)
        sampleav = np.mean(data, axis=0)
        #cov = np.mean([np.outer(v,v) for v in data], axis=0) - np.outer(sampleav, sampleav)
        cov = np.std(data, axis=0)
        self.fit = [curve_fit(fun, np.array(len(self.fflist)*self.qsqlist[lb:ub]), sample, \
        p0=np.ones(max(self.alphalist) + 1) ,sigma = cov)[0] for sample in data]
        self.fit_cv = curve_fit(fun, np.array(len(self.fflist)*self.qsqlist[lb:ub]), sampleav, \
        p0=np.ones(max(self.alphalist) + 1) ,sigma = cov)[0]
        return

    def plot(self, outfile):
        for i, name in enumerate(self.names):
            cv = np.mean(self.fflist[name], axis = 0)
            err = np.std(self.fflist[name], axis = 0)
            plt.xlabel('$q^2$')
            plt.ylabel(name)
            plt.errorbar(self.qsqlist, cv ,yerr = err, fmt='.', label='data')
            xv = np.arange(self.qsqlist[self.lb], self.qsqlist[self.ub-1]+0.1, 0.1)
            self.mBstar = np.array(len(xv)*[self.poles[i]])
            alphas = self.alphalist[self.numalpha*i:self.numalpha*(i+1)] 
            alphaval = [self.fit_cv[x] for x in alphas]
            yvfit = self.fitted(xv, *alphaval)
            yverr = np.std([self.fitted(xv,*([f[x] for x in alphas])) for f in self.fit], axis = 0)
            chisq = self.chisqdof(alphaval,cv,err,self.poles[i])

            plt.plot(xv,yvfit)
            plt.fill_between(xv, yvfit-yverr, yvfit+yverr, color='orange', alpha = 0.7, label='+'.join(['({1:.2f})$z^{0}$'.format(i,j) for i,j in enumerate(alphaval)]))
            plt.title('$\chi^2/dof = {:.2e}, residue = {}$'.format(chisq,self.poly(self.z(self.poles[i])-self.z(0),*alphaval)))
            plt.legend()
            plt.savefig('{}/{}.pdf'.format(outfile,name))
            plt.show()

    def chisqdof(self, alphaval, cv, err, pole):
        self.mBstar = np.array((self.ub-self.lb)*[pole])
        return np.sum((self.fitted(np.array(self.qsqlist[self.lb:self.ub]),*alphaval) - cv[self.lb:self.ub])**2/err[self.lb:self.ub]**2) / (self.ub-self.lb-self.numalpha)


    def covalpha(self, *other):
        if self.fit == None:
            print('Do the fit first')
            return

        if other == ():
            fit = self.fit
        else:
            fit = np.concatenate([self.fit] + [x.fit for x in list(other)] ,axis=1)

        alphaav = np.mean(fit, axis = 0)
        return np.mean([np.outer(f,f) for f in fit], axis=0) - np.outer(alphaav,alphaav)
