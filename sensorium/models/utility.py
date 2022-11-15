import copy


def prepare_grid(grid_mean_predictor, dataloaders):
    """
    Utility function for using the neurons cortical coordinates
    to guide the readout locations in image space.

    Args:
        grid_mean_predictor (dict): config dictionary, for example:
          {'type': 'cortex',
           'input_dimensions': 2,
           'hidden_layers': 1,
           'hidden_features': 30,
           'final_tanh': True}

        dataloaders: a dictionary of dataloaders, one PyTorch DataLoader per session
            in the format {'data_key': dataloader object, .. }
    Returns:
        grid_mean_predictor (dict): config dictionary
        grid_mean_predictor_type (str): type of the information that is being used for
            the grid positition estimator
        source_grids (dict): a grid of points for each data_key

    """
    if grid_mean_predictor is None:
        grid_mean_predictor_type = None
        source_grids = None
    else:
        grid_mean_predictor = copy.deepcopy(grid_mean_predictor)
        grid_mean_predictor_type = grid_mean_predictor.pop("type")

        if grid_mean_predictor_type == "cortex":
            input_dim = grid_mean_predictor.pop("input_dimensions", 2)
            source_grids = {
                k: v.dataset.neurons.cell_motor_coordinates[:, :input_dim]
                for k, v in dataloaders.items()
            }
    return grid_mean_predictor, grid_mean_predictor_type, source_grids



def purge_state_dict(state_dict, purge_key=None, survival_key=None):

    if (purge_key is None) and (survival_key is None):
        raise ValueError("purge_key and survival_key can not both be None. At least one key has to be defined")

    purged_state_dict = copy.deepcopy(state_dict)

    for dict_key in state_dict.keys():
        if (purge_key is not None) and (purge_key in dict_key):
            purged_state_dict.pop(dict_key)
        elif (survival_key is not None) and (survival_key not in dict_key):
            purged_state_dict.pop(dict_key)

    return purged_state_dict


def get_readout_key_names(model):
    data_key = list(model.readout.keys())[0]
    readout = model.readout[data_key]

    feature_name = "features"
    if "mu" in dir(readout):
        feature_name = "features"
        grid_name = "mu"
        bias_name = "bias"
    else:
        feature_name = "features"
        grid_name = "grid"
        bias_name = "bias"

    return feature_name, grid_name, bias_name