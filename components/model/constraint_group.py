'''
Class that groups constraints and form factors that they use for later use in the fitter routine
'''
from .constraint import Constraint

class ConstraintGroup:
    '''
    Class that groups constraints and form factors that they use for later use in the fitter routine
    '''
    def __init__(self):
        self.form_factor_names = set()
        self.constraints = []

    def add_form_factor(self, ff_name: str):
        '''
        Adds a form factor name to the constraint group
        '''
        self.form_factor_names.update([ff_name])

    def add_constraint(self, constr: Constraint):
        '''
        Adds a new constraint to the group
        '''
        self.constraints.append(constr)
        self.form_factor_names.update(set(constr.ff_names))
