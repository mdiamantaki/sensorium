import datajoint as dj

from .nnfabrik_tables import TrainedModel, schema, Fabrikant, Dataset, Model, Trainer, Seed
from .mei_tables import MEISeed, MEIMethod, MEISelector, TrainedEnsembleModel, MEI
from nnfabrik.utility.nnf_helper import FabrikCache

DataCache = FabrikCache(base_table=Dataset, cache_size_limit=1)
TrainedModelCache = FabrikCache(base_table=TrainedModel, cache_size_limit=1)
EnsembleModelCache = FabrikCache(base_table=TrainedEnsembleModel, cache_size_limit=1)