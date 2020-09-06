'''
Fitter module that performs form factor fits
'''
from typing import Dict, List
import numpy as np
from scipy.optimize import least_squares
from .model.data import BareFormFactor, BareFormFactorSamples
from .model.data import FittedFormFactor, FittedFormFactorSamples
from .model.constraint_group import ConstraintGroup
from .input_reader import InputReader

BIG_NUMBER = 1e10

class FitSamples:
    '''
    Fits multiple samples
    '''
    def __init__(
            self,
            bare_ff_samples: Dict[str, BareFormFactorSamples],
            constraint_group: ConstraintGroup,
            input_reader: InputReader
    ):
        errors = {ff: bare_ff_samples[ff].error for ff in bare_ff_samples}
        num_samples = len(bare_ff_samples[list(bare_ff_samples.keys())[0]].samples)
        samples = [{ff: bare_ff_samples[ff].samples[i]
                    for ff in bare_ff_samples} for i in range(num_samples)]
        self.fit_samples = [FitOne(
            sample, errors, constraint_group, input_reader) for sample in samples]

    def fit(self) -> FittedFormFactorSamples:
        '''
        Returns the list of fitted form factors for each sample
        '''
        result = [x.fit() for x in self.fit_samples]
        return FittedFormFactorSamples(result)


class FitOne:
    '''
    Fitter class - performs fitting routine on a single sample
    '''
    def __init__(
            self,
            bare_form_factors: Dict[str, BareFormFactor],
            bare_ff_errors: Dict[str, List[float]],
            constraint_group: ConstraintGroup,
            input_reader: InputReader
    ):
        self.ff_names = constraint_group.form_factor_names
        self.constraint_group = constraint_group
        self.bare_form_factors = {ff: bare_form_factors[ff] for ff in self.ff_names}
        self.bare_ff_errors = {ff: np.array(bare_ff_errors[ff]) for ff in self.ff_names}
        self.input_reader = input_reader
        ff_templates = input_reader.create_formfactor_templates()
        self.fitted_form_factors = {ff: ff_templates[ff] for ff in self.ff_names}

    def fit(self) -> Dict[str, FittedFormFactor]:
        '''
        Performs the fit and returns the fitted form factors
        '''
        initial_guess = np.ones(np.sum(
            [len(self.fitted_form_factors[ff].params) for ff in self.fitted_form_factors]))
        solution = least_squares(self._fit_residues, initial_guess).x
        self._distribute_parameters(solution)
        return self.fitted_form_factors

    def _fit_residues(self, parameters) -> List[float]:
        '''
        Helper function to calculate the residues of a fit
        (not to be confused with residues of the form factor)
        '''
        self._distribute_parameters(parameters)
        result = []
        for form_factor in self.ff_names:
            lower, upper = self.get_bounds(form_factor)
            qsqlist = self.bare_form_factors[form_factor].qsqlist[lower:upper]
            fitted_ff_values = np.array(
                [self.fitted_form_factors[form_factor].eval(qsq) for qsq in qsqlist])
            bare_ff_values = np.array(self.bare_form_factors[form_factor].values[lower:upper])
            result.extend((fitted_ff_values - bare_ff_values) /
                          self.bare_ff_errors[form_factor][lower:upper])

        for constr in self.constraint_group.constraints:
            result.append(BIG_NUMBER*constr.eval(self.fitted_form_factors))

        return result

    def get_bounds(self, ff_name: str) -> List[int]:
        '''
        Converts the fit bounds in GeV^2 to index numbers
        '''
        lower, upper = self.input_reader.get_bounds(ff_name)
        qsqlist = self.bare_form_factors[ff_name].qsqlist
        for i, qsq in enumerate(qsqlist):
            if qsq >= lower:
                lower_bound = i
                break
        for i, qsq in reversed(list(enumerate(qsqlist))):
            if qsq <= upper:
                upper_bound = i+1
                break
        return [lower_bound, upper_bound]

    def _distribute_parameters(self, parameters):
        '''
        Distributes fit parameters among form factors
        '''
        num_parameters = [len(self.fitted_form_factors[ff].params) for ff in self.ff_names]
        splits = np.cumsum(num_parameters)
        split_parameters = np.split(parameters, splits)
        for form_factor, ff_params in zip(self.ff_names, split_parameters):
            self.fitted_form_factors[form_factor].set_params(ff_params)
