from nnfabrik.main import my_nnfabrik
import datajoint as dj


main_nnfabrik = my_nnfabrik(
    schema=dj.config['nnfabrik.schema_name'],
    use_common_fabrikant=False,
)

Fabrikant, Dataset, Model, Trainer, Seed = map(
    main_nnfabrik.__dict__.get, ["Fabrikant", "Dataset", "Model", "Trainer", "Seed"]
)

schema = main_nnfabrik.schema

import os

from nnfabrik.templates.trained_model import TrainedModelBase
from nnfabrik.utility.dj_helpers import CustomSchema, make_hash, cleanup_numpy_scalar


@schema
class TrainedModel(TrainedModelBase):
    table_comment = "Trained models"
    nnfabrik = main_nnfabrik
    data_info_table = None 


@schema
class SeedHyper(dj.Manual):
    definition = """
    seed:   int     # Random seed that is passed to the model- and dataset-builder
    """
    

@schema
class TrainedHyperModel(TrainedModelBase):
    nnfabrik = main_nnfabrik
    seed_table = SeedHyper # add the seed table here
    data_info_table = None 