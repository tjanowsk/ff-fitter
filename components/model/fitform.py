'''
Classes that encode different fit forms.
'''
import abc
from typing import List
import numpy as np

class FitForm:
    '''
    Defines a generic fit form template
    '''
    @abc.abstractmethod
    def call(self, qsq: float, params: List[float]) -> float:
        '''
        Evaluates the function at a given q^2 with given set of params
        '''

    @abc.abstractmethod
    def residue(self, params: List[float]) -> float:
        '''
        Returns the residue given the set of params
        '''

class FitFormZExp(FitForm):
    '''
    z-expansion
    '''
    def __init__(self, mB: float, mV: float, mpole: float, num_params: int):
        self.mB = float(mB)
        self.mV = float(mV)
        self.mpole = float(mpole)
        self.num_params = int(num_params)
        self.tplus = (self.mB+self.mV)**2
        self.tminus = (self.mB-self.mV)**2
        self.t0 = self.tplus*(1-np.sqrt(1-self.tminus/self.tplus))
        self.roottpt0 = np.sqrt(self.tplus - self.t0)
        self.z_0 = self._z(0)

    def call(self, qsq: float, params: List[float]) -> float:
        if len(params) != self.num_params:
            raise ValueError(
                f"Wrong number of parameters, expected {self.num_params} got {len(params)}")
        return 1.0/(1.0 - qsq/self.mpole**2) * self._poly(qsq, params)

    def residue(self, params: List[float]) -> float:
        return -self.mpole**2 * self._poly(self.mpole**2, params)

    def _poly(self, qsq, params) -> float:
        result = 0
        for n, param in enumerate(params):
            result += param*(self._z(qsq) - self.z_0)**n
        return result

    def _z(self, qsq: float) -> float:
        root_tpmq2 = np.sqrt(self.tplus - qsq)
        return (root_tpmq2 - self.roottpt0)/(root_tpmq2 + self.roottpt0)