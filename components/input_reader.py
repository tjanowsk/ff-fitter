'''
Classes for reading the input json file
'''
from typing import Dict, List
import abc
import json
import numpy as np
from .model.constraint import Constraint
from .model.data import FittedFormFactor
from .model.fitform import fit_form_factory

class InputReader:
    '''
    Abstract reader class defining methods that must be defined for any reader class
    '''
    @abc.abstractmethod
    def get_input(self):
        '''
        Returns the contents of the input file
        '''

    @abc.abstractmethod
    def get_ff_names(self):
        '''
        Returns names of all form factors in the input
        '''

    @abc.abstractmethod
    def create_formfactor_templates(self) -> Dict[str, FittedFormFactor]:
        '''
        Creates empty templates for fitted form factors to be passed to a fitter routine
        '''

    @abc.abstractmethod
    def get_constraints(self) -> List[Constraint]:
        '''
        Returns the constraints
        '''

    @abc.abstractmethod
    def get_bounds(self, ff_name: str):
        '''
        Returns fit bounds for a given form factor
        '''

class JsonInputReader(InputReader):
    '''
    Reads the input json file
    '''
    def __init__(self, input_file_name: str):
        with open(input_file_name, 'r') as file:
            self.input = json.load(file)

    def get_input(self):
        return self.input

    def create_formfactor_templates(self) -> Dict[str, FittedFormFactor]:
        '''
        Creates empty templates for fitted form factors to be passed to a fitter routine
        '''
        filenames = [filename for filename in self.input if filename != 'constraints']
        result = {}
        for filename in filenames:
            for form_factor in self.input[filename]:
                input_parameters = self.input[filename][form_factor]
                fit_form = fit_form_factory(
                    input_parameters['fit_form'],
                    input_parameters
                )
                result[form_factor] = FittedFormFactor(
                    fit_form, np.ones(input_parameters['num_pars']))
        return result

    def get_ff_data(self, ff_name: str):
        '''
        Returns data for a given form factor
        '''
        filenames = [filename for filename in self.input if filename != 'constraints']
        for filename in filenames:
            if ff_name in self.input[filename]:
                return self.input[filename][ff_name]

    def get_ff_names(self) -> List[str]:
        filenames = [filename for filename in self.input if filename != 'constraints']
        result = []
        for filename in filenames:
            result.extend(self.input[filename].keys())
        return result

    def get_bounds(self, ff_name: str) -> List[float]:
        '''
        Returns fit bounds for a given form factor in GeV^2
        '''
        form_factor = self.get_ff_data(ff_name)
        return [form_factor['lb'], form_factor['ub']]

    def get_constraints(self) -> List[Constraint]:
        '''
        Returns the constraints
        '''
        return [Constraint(constr_string) for constr_string in self.input['constraints']]
