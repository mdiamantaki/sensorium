import datajoint as dj

from . import *
from .templates import ScoringTable
from ..utility.scores import get_signal_correlations, get_correlations


@schema
class CorrelationToAverage(ScoringTable):
    trainedmodel_table = TrainedModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_signal_correlations)
    measure_dataset = "test"
    measure_attribute = "avg_correlation"
    data_cache = None
    model_cache = TrainedModelCache
    function_kwargs = dict(tier="test")
    

@schema
class ValidationCorrelationToSingle(ScoringTable):
    trainedmodel_table = TrainedModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_correlations)
    measure_dataset = "validation"
    measure_attribute = "single_correlation"
    data_cache = DataCache
    model_cache = TrainedModelCache


@schema
class SingleTrialCorrelationTest(ScoringTable):
    trainedmodel_table = TrainedModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_correlations)
    measure_dataset = "test"
    measure_attribute = "single_correlation"
    data_cache = None
    model_cache = TrainedModelCache
    function_kwargs = dict(tier="test")