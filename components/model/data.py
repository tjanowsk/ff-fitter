'''
Model classes defining data structures used in other functions
'''
from typing import Dict, List
import numpy as np
from .fitform import FitForm

class BareFormFactor:
    '''
    Class containing bare form factor data
    '''
    def __init__(self, qsqlist: List[float], values: List[float]):
        self.qsqlist = qsqlist
        self.values = np.array(values)

    def __repr__(self):
        return str(self.values)

class BareFormFactorSamples:
    '''
    Class containing all samples for a given bare form factor
    '''
    def __init__(self, samples: List[BareFormFactor]):
        self.samples = samples
        value_list = [sample.values for sample in samples]
        self.mean = np.mean(value_list, axis=0)
        self.error = np.std(value_list, axis=0)

    def __getitem__(self, key):
        return self.samples[key]

    def get_qsqlist(self):
        '''
        Returns q^2 list for a given bare form factor
        '''
        return self.samples[0].qsqlist

class FittedFormFactor:
    '''
    Class containing fitted parameters of a given form factor
    '''
    def __init__(self, fit_form: FitForm, params: List[float]):
        self.fit_form = fit_form
        self.params = params

    def set_params(self, params: List[float]):
        '''
        Sets the fit parameters, used by the fit function
        '''
        self.params = params

    def eval(self, qsq: float) -> float:
        '''
        Calculate the value of the fitted form factor at a given q^2
        '''
        return self.fit_form.call(qsq, self.params)

    def residue(self) -> float:
        '''
        Calculates the residue of the fitted form factor
        '''
        return self.fit_form.residue(self.params)

class FittedFormFactorSamples:
    '''
    Contains samples of fitted form factors, result of the fitting procedure
    '''
    def __init__(self, samples: List[Dict[str, FittedFormFactor]]):
        self.samples = {ff: [sample[ff]
                             for sample in samples] for ff in samples[0]}

    def mean(self, form_factor: str) -> List[float]:
        '''
        Returns a mean of one particular form factor
        '''
        return np.mean([ff.params for ff in self.samples[form_factor]], axis=0)

    def full_mean(self) -> Dict[str, List[float]]:
        '''
        Returns a mean for all form factors
        '''
        return {ff: self.mean(ff) for ff in self.samples}

    def covariance(self, form_factor1: str, form_factor2: str) -> List[List[float]]:
        '''
        Returns the covariance matrix between a pair of form factors
        '''
        samples_iterator = zip(
            self.samples[form_factor1], self.samples[form_factor2])
        ff1_ff2 = np.mean([np.outer(ff1.params, ff2.params)
                           for ff1, ff2 in samples_iterator], axis=0)
        return ff1_ff2 - np.outer(self.mean(form_factor1), self.mean(form_factor2))

    def full_covariance(self) -> Dict[str, Dict[str, List[List[float]]]]:
        '''
        Returns full covariance matrix
        '''
        result = {}
        for ff1 in self.samples:
            result[ff1] = {}
            for ff2 in self.samples:
                result[ff1][ff2] = self.covariance(ff1, ff2)
        return result
