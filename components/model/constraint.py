'''
Class that defines a fit constraint
'''
from typing import Dict, List
import re
from .data import BareFormFactor, FittedFormFactor

class Constraint:
    '''
    Class that defines a fit constraint
    '''
    def __init__(self, constraint_expr: str):
        self.constraint_expr = re.sub(
            r'([0-9A-Za-z_-]+)\(([0-9A-Za-z_*.-]+)\)',
            r'form_factors["\g<1>"].eval(\g<2>)',
            constraint_expr)
        self.constraint_expr = re.sub(
            r'<([0-9A-Za-z_-]+)>',
            r'form_factors["\g<1>"].residue()',
            self.constraint_expr)
        self.constraint_expr = re.sub(
            r'\[([0-9A-Za-z_-]+)\]',
            r'bare_ff["\g<1>"].residue',
            self.constraint_expr)
        self.ff_names = _get_form_factor_names(constraint_expr)

    def __repr__(self):
        return self.constraint_expr

    def eval(
            self,
            form_factors: Dict[str, FittedFormFactor],
            bare_ff: Dict[str, BareFormFactor]
    ) -> float:
        '''
        Evaluates the constraint
        '''
        return eval(self.constraint_expr)

def _get_form_factor_names(
        constraint_expr: str
) -> List[str]:
    '''
    Helper function that returns all form factor names that are a part of the constraint
    '''
    pat = re.compile(r'[a-zA-Z0-9_-]+(?=\([a-zA-Z0-9_.*-]+\))')
    pat2 = re.compile(r'(?<=<)[a-zA-Z0-9_-]+(?=>)')
    form_factor_names = list(set(pat.findall(constraint_expr) + pat2.findall(constraint_expr)))
    return form_factor_names
