import numpy as np
import h5py
from scipy.optimize import least_squares
from matplotlib import pyplot as plt
import json, re
from itertools import cycle

class FormFactor:
    def __init__(self, qsqlist, samples, mB, mpole, fitForm, numParams, lb, ub, mV=0.77):
        self.qsqlist = np.array(qsqlist)
        self.samples = np.array(samples)
        self.lb = lb
        self.ub = ub
        self.mB = mB
        self.mV = mV
        self.mpole = mpole
        self.fitForm = fitForm
        self.numParams = numParams
        self.errors = None
    def calculateErrors(self):
        self.errors = np.std(self.samples, axis=0, dtype=np.float64)
    def calculateResidue(self, sampleNumber, parameters):
        if self.errors is None:
            self.calculateErrors()
        result = (self.function(self.qsqlist[self.lb:self.ub], parameters) - self.samples[sampleNumber, self.lb:self.ub])/self.errors[self.lb:self.ub]
        return result
    def z(self, qsq):
        tplus = (self.mB+self.mV)**2
        tminus =(self.mB-self.mV)**2
        t0 = tplus*(1-np.sqrt(1-tminus/tplus))
        return (np.sqrt(tplus-qsq)-np.sqrt(tplus-t0))/(np.sqrt(tplus-qsq) + np.sqrt(tplus-t0))
    def poly(self, parameters, x):
        res = 0
        for n, a in enumerate(parameters):
            res += a*x**n
        return res
    def function(self, qsq, parameters):
        return 1.0/(1.0 - qsq/self.mpole**2) * self.poly(parameters, self.z(qsq) - self.z(0))
    def residue(self, parameters):
        return -self.mpole**2 * self.poly(parameters, self.z(self.mpole**2) - self.z(0))

class Fitter:
    def __init__(self, jsonFilename):
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
        with open(jsonFilename, "r") as f:
            input = json.load(f)

        self.constraints = input["constraints"]
        self.formFactors = {}
        for dataFilename in input:
            if dataFilename == "constraints":
                continue
            with h5py.File(dataFilename, 'r') as f:
                qsqlist = list(f['qsqlist'][...])
                data = {key: f[key][...] for key in f}
                for key in input[dataFilename]:
                    mB = np.float(input[dataFilename][key]["mB"])
                    mpole = np.float(input[dataFilename][key]["m_pole"])
                    numParams = np.int(input[dataFilename][key]["num_pars"])
                    fitForm = input[dataFilename][key]["fit_form"]
                    lb, ub = getbounds(qsqlist, np.int(input[dataFilename][key]["lb"]), np.int(input[dataFilename][key]["ub"]))
                    self.formFactors[key] = FormFactor(qsqlist, data[key], mB, mpole, fitForm, numParams, lb, ub)
        self.NumberOfSamples = len(self.formFactors[list(self.formFactors.keys())[0]].samples)
        self.sampleNumber = 0
        self.fit = None

    def constraintsToSets(self):
        pat = re.compile("[a-zA-Z0-9_-]+(?=\([a-zA-Z0-9_.*-]+\))")
        pat2 = re.compile("(?<=<)[a-zA-Z0-9_-]+(?=>)")
        return [ [set(pat.findall(c)),c] for c in self.constraints if pat.findall(c) != []] +\
                 [ [set(pat2.findall(c)),c] for c in self.constraints if pat2.findall(c) != []]

    def maximalSets(self):
        setList = self.constraintsToSets()
        result = []
        while setList != []:
            ff, constr = setList.pop()
            constr = [constr,]
            for ff2, c2 in reversed(setList):
                if ff.intersection(ff2) != set():
                    ff = ff.union(ff2)
                    constr.append(c2)
                    setList.remove([ff2,c2])
            result.append([ff, constr])
        return [[list(r[0]), list(r[1])] for r in result]
    
    def splitParameters(self, parameters):
        splits = np.cumsum([self.formFactors[ff].numParams for ff in self.partialfflist], axis=-1)
        if len(self.partialfflist) == 1:
            return {self.partialfflist[0]: parameters}
        parameterSplits = np.split(parameters, splits, axis=-1)
        return {key:item for key,item in zip(self.partialfflist, parameterSplits)}

    def evalConstraints(self, paramDict):
        BIGNUMBER = 1e8
        res = []
        for constraint in self.partialconstrs:
            constraint = re.sub("([0-9A-Za-z_-]+)\(([0-9A-Za-z_*.-]+)\)", "self.formFactors[\"\g<1>\"].function(\g<2>, paramDict[\"\g<1>\"])", constraint)
            constraint = re.sub("<([0-9A-Za-z_-]+)>", "self.formFactors[\"\g<1>\"].residue(paramDict[\"\g<1>\"])", constraint)
            res.append(BIGNUMBER*(eval(constraint)))
        return np.array(res)

    def calculateResidue(self, parameters):
        paramDict = self.splitParameters(parameters)
        result = np.concatenate([self.formFactors[ff].calculateResidue(self.sampleNumber, paramDict[ff]) for ff in self.partialfflist] +
                [self.evalConstraints(paramDict)])
        return result

    def generateFit(self, verbose=True):
        maxSets = self.maximalSets()
        constrainedFFs = np.concatenate([ff for ff,c in maxSets])
        for ff in self.formFactors:
            if (ff not in constrainedFFs):
                maxSets.append([[ff],[]])
        self.fit = {key:[] for key in self.formFactors}

        for self.partialfflist, self.partialconstrs in maxSets:
            if(verbose):
                print("Now fitting FFs:\n",self.partialfflist, "\n subject to constraints: \n", self.partialconstrs)
            for self.sampleNumber in range(self.NumberOfSamples):
                p0 = np.ones(np.sum([self.formFactors[ff].numParams for ff in self.partialfflist]))
                fittedPars = self.splitParameters(least_squares(self.calculateResidue, p0).x)
                for key in fittedPars:
                    self.fit[key].append(fittedPars[key])

    def meanFitParameters(self):
        return {key: np.mean(self.fit[key], axis=0) for key in self.fit}

    def covarianceMatrix(self):
        def cov(samples1, samples2):
            sampleAverage1 = np.mean(samples1, axis=0)
            sampleAverage2 = np.mean(samples2, axis=0)
            return np.mean([np.outer(x, y) for x, y in zip(samples1, samples2)], axis=0) - np.outer(sampleAverage1, sampleAverage2)
        return {name1: {name2: cov(self.fit[name1], self.fit[name2]) for name2 in self.fit} for name1 in self.fit}

    def getResidues(self):
        residues = {name:[self.formFactors[name].residue(sample) for sample in self.fit[name]] for name in self.formFactors}
        return {name: (np.mean(residues[name]), np.std(residues[name])) for name in residues}

    def plot(self, fflist):
        colorCycler = cycle("rbgcmyk")
        for i, name in enumerate(fflist):
            color = colorCycler.__next__()

            cv = np.mean(self.formFactors[name].samples, axis=0)
            err = np.std(self.formFactors[name].samples, axis=0)
            plt.xlabel('$q^2$')

            ff = self.formFactors[name]
            qsqlist = np.array(self.formFactors[name].qsqlist)
            lb = self.formFactors[name].lb
            ub = self.formFactors[name].ub

            plt.errorbar(qsqlist, cv, yerr=err, fmt='.', label=name, color=color)
            xv = np.arange(qsqlist[lb], qsqlist[ub-1]+0.1, 0.1)
            xv_below = np.arange(min(qsqlist),qsqlist[lb] + 0.1, 0.1) 
            xv_above = np.arange(qsqlist[ub-1] + 0.1, max(qsqlist) + 0.1, 0.1)

            #alphadict = {n:params for n, params in zip(self.formFactors, self.splitParameters(self.fit))}
            yvfit = ff.function(xv, np.mean(self.fit[name], axis=0))
            yvfit_below = ff.function(xv_below, np.mean(self.fit[name], axis=0))
            yvfit_above = ff.function(xv_above, np.mean(self.fit[name], axis=0))

            yverr = np.std([ff.function(xv, sample) for sample in self.fit[name]], axis=0)
            yverr_below = np.std([ff.function(xv_below, sample) for sample in self.fit[name]], axis=0)
            yverr_above = np.std([ff.function(xv_above, sample) for sample in self.fit[name]], axis=0)

            plt.plot(xv, yvfit, color=color)
            plt.plot(xv_below, yvfit_below, '--', color=color)
            plt.plot(xv_above, yvfit_above, '--', color=color)
            plt.fill_between(xv, yvfit-yverr, yvfit+yverr, color=color, alpha=0.6)
            plt.fill_between(xv_below, yvfit_below-yverr_below, yvfit_below+yverr_below, color=color, alpha=0.3)
            plt.fill_between(xv_above, yvfit_above-yverr_above, yvfit_above+yverr_above, color=color, alpha=0.3)
        plt.legend()
        plt.show()
