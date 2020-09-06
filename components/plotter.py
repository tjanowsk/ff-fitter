'''
Form factor plotter module
'''
from typing import Dict, List
from itertools import cycle
from matplotlib import pyplot as plt
import numpy as np
from .model.data import BareFormFactorSamples, FittedFormFactorSamples

class Plotter:
    '''
    Plotter class - plotting form factors
    '''
    def __init__(
            self,
            bare_form_factors: Dict[str, BareFormFactorSamples],
            fitted_form_factors: List[FittedFormFactorSamples]
    ):
        self.bare_form_factors = bare_form_factors
        self.fitted_form_factors = fitted_form_factors
        self.color_cycles = cycle('rgbcmyk')

    def add_form_factor(self, ff_name: str, xval: List[float], **kwargs):
        '''
        Adds a form factor data to the plot
        '''
        if 'color' not in kwargs:
            color = self.color_cycles.__next__()
        else:
            color = kwargs['color']

        if 'label' not in kwargs:
            label = ff_name
        else:
            label = kwargs['label']

        self.add_data_plot(ff_name, color=color, label=label)
        self.add_fitted_curve(ff_name, xval, color=color, label='')

    def add_data_plot(self, ff_name: str, **kwargs):
        '''
        Plots raw form factor data as points
        '''
        xval = self.bare_form_factors[ff_name].get_qsqlist()
        yval = self.bare_form_factors[ff_name].mean
        errval = self.bare_form_factors[ff_name].error

        plt.errorbar(xval, yval, errval, fmt='.', **kwargs)

    def add_fitted_curve(self, ff_name: str, xrange: List[float], **kwargs):
        '''
        Plots fitted form factor data as a curve with error bands
        '''
        for fit_group in self.fitted_form_factors:
            if ff_name in fit_group.samples:
                ff = fit_group.samples[ff_name]
        yvals = [[sample.eval(x) for x in xrange] for sample in ff]
        mean_yval = np.mean(yvals, axis=0)
        err_yval = np.std(yvals, axis=0)
        plt.plot(xrange, mean_yval, **kwargs)
        plt.fill_between(xrange, mean_yval - err_yval, mean_yval + err_yval, alpha=0.3, **kwargs)

    def show(self):
        '''
        Displays the plot
        '''
        plt.legend()
        plt.show()
