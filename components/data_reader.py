'''
Form factor data reader module
'''
import abc
import h5py
from .input_reader import InputReader
from .model.data import BareFormFactor, BareFormFactorSamples

class DataReader:
    '''
    Generic data reader interface
    '''
    @abc.abstractmethod
    def get_bare_ff_samples(self, ff_name: str) -> BareFormFactorSamples:
        '''
        Returns bare form factor samples
        '''

class H5DataReader(DataReader):
    '''
    HDF5 data reader
    '''
    def __init__(self, input_file: InputReader):
        self.ff_data = {}
        input_contents = input_file.get_input()
        for filename in input_contents:
            if filename == 'constraints':
                continue
            with h5py.File(filename, 'r') as datafile:
                for ff_name in input_contents[filename]:
                    form_factor_samples = [
                        BareFormFactor(datafile['qsqlist'][...],
                                       sample) for sample in datafile[ff_name][...]]
                    self.ff_data[ff_name] = BareFormFactorSamples(
                        form_factor_samples)

    def get_bare_ff_samples(self, ff_name: str) -> BareFormFactorSamples:
        return self.ff_data[ff_name]
