from __future__ import annotations
from typing import Dict, Any

import datajoint as dj
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

Key = Dict[str, Any]
Dataloaders = Dict[str, DataLoader]


from mei import mixins
from mei.main import MEISeed, MEIMethod
from mei.modules import ConstrainedOutputModel
from .nnfabrik_tables import TrainedModel, schema, Fabrikant, Dataset, Model, Trainer, Seed


class MouseSelectorTemplate(dj.Computed):

    dataset_table = Dataset
    definition = """
    # contains information that can be used to map a neuron's id to its corresponding integer position in the output of
    # the model. 
    -> self.dataset_table
    unit_id       : int               # unique neuron identifier
    data_key      : varchar(255)      # unique session identifier
    ---
    unit_index : int                    # integer position of the neuron in the model's output 
    """

    constrained_output_model = ConstrainedOutputModel

    def make(self, key):
        dataloaders = (Dataset & key).get_dataloader()
        data_keys = list(dataloaders["train"].keys())

        mappings = []
        for data_key in data_keys:
            dat = dataloaders["train"][data_key].dataset
            try:
                neuron_ids = dat.neurons.unit_ids
            except AttributeError:
                warnings.warn(
                    "unit_ids were not found in the dataset - using indices 0-N instead"
                )
                neuron_ids = range(dat.responses.shape[1])
            for neuron_pos, neuron_id in enumerate(neuron_ids):
                mappings.append(
                    dict(
                        key, unit_id=neuron_id, unit_index=neuron_pos, data_key=data_key
                    )
                )

        self.insert(mappings)

    def get_output_selected_model(
        self, model: Module, key: Key
    ) -> constrained_output_model:
        unit_index, data_key = (self & key).fetch1("unit_index", "data_key")
        return self.constrained_output_model(
            model, unit_index, forward_kwargs=dict(data_key=data_key)
        )


@schema
class MEISelector(MouseSelectorTemplate):
    dataset_table = Dataset
    

@schema
class TrainedEnsembleModel(mixins.TrainedEnsembleModelTemplateMixin, dj.Manual):
    dataset_table = Dataset
    trained_model_table = TrainedModel

    class Member(mixins.TrainedEnsembleModelTemplateMixin.Member, dj.Part):
        pass
    


@schema
class MEI(mixins.MEITemplateMixin, dj.Computed):
    """MEI table template.

    To create a functional "MEI" table, create a new class that inherits from this template and decorate it with your
    preferred Datajoint schema. Next assign your trained model (or trained ensemble model) and your selector table to
    the class variables called "trained_model_table" and "selector_table". By default, the created table will point to
    the "MEIMethod" table in the Datajoint schema called "nnfabrik.main". This behavior can be changed by overwriting
    the class attribute called "method_table".
    """

    trained_model_table = TrainedEnsembleModel
    selector_table = MEISelector
    method_table = MEIMethod
    seed_table = MEISeed
