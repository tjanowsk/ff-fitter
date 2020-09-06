'''
Analysis module
'''
from typing import List
import numpy as np
from .model.constraint import Constraint
from .model.constraint_group import ConstraintGroup
from .fitter import FitSamples
from .input_reader import InputReader
from .data_reader import DataReader

class Analysis:
    '''
    Form factor analysis class
    '''
    def __init__(self, input_file_reader: InputReader, data_reader: DataReader):
        self.bare_form_factors = {ff_name: data_reader.get_bare_ff_samples(
            ff_name) for ff_name in input_file_reader.get_ff_names()}
        self.fit_groups = _group_constraints(input_file_reader.get_constraints())
        included_form_factors = np.concatenate(
            [list(c.form_factor_names) for c in self.fit_groups])
        for ff in self.bare_form_factors:
            if ff not in included_form_factors:
                fit_group = ConstraintGroup()
                fit_group.add_form_factor(ff)
                self.fit_groups.append(fit_group)
        for fit_group in self.fit_groups:
            print(fit_group.constraints, fit_group.form_factor_names)
        self.fitters = [FitSamples(
            self.bare_form_factors, fit_group, input_file_reader) for fit_group in self.fit_groups]

    def fit(self):
        '''
        perform the fits for all chi^2 groups
        '''
        return [x.fit() for x in self.fitters]

def _group_constraints(constraints: List[Constraint]) -> List[ConstraintGroup]:
    constraint_groups = []
    constr_list = [c for c in constraints]
    while constr_list != []:
        constr = constr_list.pop()
        ff = set(constr.ff_names)
        constraint_group = ConstraintGroup()
        constraint_group.add_constraint(constr)

        for constr2 in reversed(constr_list):
            ff2 = constr2.ff_names
            if set(ff2).intersection(ff) != set():
                ff = ff.union(ff2)
                constraint_group.add_constraint(constr2)
                constr_list.remove(constr2)
        constraint_groups.append(constraint_group)
    return constraint_groups
