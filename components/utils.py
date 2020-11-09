from typing import List
from .model.data import FittedFormFactorSamples

def combine_results(result_list: List[FittedFormFactorSamples]) -> FittedFormFactorSamples:
    combined_ff_list = {}
    for fitted_ff_list in result_list:
        combined_ff_list.update(fitted_ff_list.samples)

    result = FittedFormFactorSamples([{}])
    result.samples = combined_ff_list
    return result
