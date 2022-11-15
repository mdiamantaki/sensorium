from torch import nn

from nnfabrik.utility.nn_helpers import set_random_seed, get_dims_for_loader_dict
from neuralpredictors.utils import get_module_output
from neuralpredictors.layers.encoders import FiringRateEncoder
from neuralpredictors.layers.shifters import MLPShifter, StaticAffine2dShifter
from neuralpredictors.layers.cores import (
    Stacked2dCore,
    SE2dCore,
    RotationEquivariant2dCore,
)

from .readouts import MultipleFullGaussian2d
from .utility import purge_state_dict, get_readout_key_names

from ..schemas import TrainedModel, Model


class Encoder(FiringRateEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        *args,
        targets=None,
        data_key=None,
        behavior=None,
        pupil_center=None,
        trial_idx=None,
        shift=None,
        detach_core=False,
        **kwargs
    ):
        x = self.core(args[0])
        if detach_core:
            x = x.detach()

        if self.shifter and pupil_center is not None:
            shift = self.shifter[data_key](pupil_center, trial_idx)

        x = self.readout(x, data_key=data_key, shift=shift, **kwargs)

        if self.modulator and behavior is not None:
            x = self.modulator[data_key](x, behavior=behavior)

        return nn.functional.elu(x + self.offset) + 1


def stacked_core_full_gauss_readout(
    dataloaders,
    seed,
    hidden_channels=32,
    input_kern=13,
    hidden_kern=3,
    layers=3,
    gamma_input=15.5,
    skip=0,
    final_nonlinearity=True,
    momentum=0.9,
    pad_input=False,
    batch_norm=True,
    hidden_dilation=1,
    laplace_padding=None,
    input_regularizer="LaplaceL2norm",
    init_mu_range=0.2,
    init_sigma=1.0,
    readout_bias=True,
    gamma_readout=4,
    elu_offset=0,
    stack=None,
    depth_separable=False,
    linear=False,
    gauss_type="full",
    grid_mean_predictor=None,
    attention_conv=False,
    shifter=None,
    shifter_type="MLP",
    input_channels_shifter=2,
    hidden_channels_shifter=5,
    shift_layers=3,
    gamma_shifter=0,
    shifter_bias=True,
    hidden_padding=None,
    core_bias=False,
):
    """
    Model class of a stacked2dCore (from neuralpredictors) and a pointpooled (spatial transformer) readout

    Args:
        dataloaders: a dictionary of dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: random seed
        elu_offset: Offset for the output non-linearity [F.elu(x + self.offset)]
        isotropic: whether the Gaussian readout should use isotropic Gaussians or not
        grid_mean_predictor: if not None, needs to be a dictionary of the form
            {
            'type': 'cortex',
            'input_dimensions': 2,
            'hidden_layers':0,
            'hidden_features':20,
            'final_tanh': False,
            }
            In that case the datasets need to have the property `neurons.cell_motor_coordinates`
        share_features: whether to share features between readouts. This requires that the datasets
            have the properties `neurons.multi_match_id` which are used for matching. Every dataset
            has to have all these ids and cannot have any more.
        all other args: See Documentation of Stacked2dCore in neuralpredictors.layers.cores and
            PointPooled2D in neuralpredictors.layers.readouts

    Returns: An initialized model which consists of model.core and model.readout
    """

    if "train" in dataloaders.keys():
        dataloaders = dataloaders["train"]

    # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
    batch = next(iter(list(dataloaders.values())[0]))
    in_name, out_name = (
        list(batch.keys())[:2] if isinstance(batch, dict) else batch._fields[:2]
    )

    session_shape_dict = get_dims_for_loader_dict(dataloaders)
    n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
    input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = (
        list(input_channels.values())[0]
        if isinstance(input_channels, dict)
        else input_channels[0]
    )
    set_random_seed(seed)

    core = SE2dCore(
        input_channels=core_input_channels,
        hidden_channels=hidden_channels,
        input_kern=input_kern,
        hidden_kern=hidden_kern,
        layers=layers,
        gamma_input=gamma_input,
        skip=skip,
        final_nonlinearity=final_nonlinearity,
        bias=core_bias,
        momentum=momentum,
        pad_input=pad_input,
        batch_norm=batch_norm,
        hidden_dilation=hidden_dilation,
        laplace_padding=laplace_padding,
        input_regularizer=input_regularizer,
        stack=stack,
        depth_separable=depth_separable,
        linear=linear,
        attention_conv=attention_conv,
        hidden_padding=hidden_padding,
    )

    in_shapes_dict = {
        k: get_module_output(core, v[in_name])[1:]
        for k, v in session_shape_dict.items()
    }

    readout = MultipleFullGaussian2d(
        in_shape_dict=in_shapes_dict,
        loader=dataloaders,
        n_neurons_dict=n_neurons_dict,
        init_mu_range=init_mu_range,
        bias=readout_bias,
        init_sigma=init_sigma,
        gamma_readout=gamma_readout,
        gauss_type=gauss_type,
        grid_mean_predictor=grid_mean_predictor,
    )

    if shifter is True:
        data_keys = [i for i in dataloaders.keys()]
        if shifter_type == "MLP":
            shifter = MLPShifter(
                data_keys=data_keys,
                input_channels=input_channels_shifter,
                hidden_channels_shifter=hidden_channels_shifter,
                shift_layers=shift_layers,
                gamma_shifter=gamma_shifter,
            )

        elif shifter_type == "StaticAffine":
            shifter = StaticAffine2dShifter(
                data_keys=data_keys,
                input_channels=input_channels_shifter,
                bias=shifter_bias,
                gamma_shifter=gamma_shifter,
            )

    model = Encoder(
        core=core,
        readout=readout,
        shifter=shifter,
        elu_offset=elu_offset,
    )

    return model


def simple_core_transfer(dataloaders,
                         seed,
                         transfer_key=dict(),
                         core_transfer_table=None,
                         readout_transfer_table=None,
                         readout_transfer_key=dict(),
                         pretrained_features=False,
                         pretrained_grid=False,
                         pretrained_bias=False,
                         freeze_core=True,
                         data_info=None,
                         **kwargs):

    if not readout_transfer_key and (pretrained_features or pretrained_grid or pretrained_bias):
        raise ValueError("if pretrained features, positions, or bias should be transferred, a readout transfer key "
                         "has to be provided, by passing it to the argument 'readout_transfer_key'")

    # set default values that are in line with parameter expansion
    if core_transfer_table is None:
        core_transfer_table = TrainedTransferModel
    elif core_transfer_table == "TrainedModel":
        core_transfer_table = TrainedModel

    if readout_transfer_table is None:
        readout_transfer_table = TrainedModel

    if kwargs:
        model_fn, model_config = (Model & transfer_key).fetch1("model_fn", "model_config")
        model_config.update(kwargs)
        model = get_model(model_fn=model_fn,
                          model_config=model_config,
                          dataloaders=dataloaders,
                          seed=seed)
    else:
        model = (Model & transfer_key).build_model(dataloaders=dataloaders, seed=seed, data_info=data_info)
    model_state = (core_transfer_table & transfer_key).get_full_config(include_state_dict=True)["state_dict"]

    core = purge_state_dict(state_dict=model_state, purge_key='readout')
    model.load_state_dict(core, strict=False)

    if freeze_core:
        for params in model.core.parameters():
            params.requires_grad = False

    if readout_transfer_key:
        readout_state = (readout_transfer_table & readout_transfer_key).get_full_config(include_state_dict=True)["state_dict"]
        readout = purge_state_dict(state_dict=readout_state, purge_key='core')
        feature_key, grid_key, bias_key = get_readout_key_names(model)

        if not pretrained_features:
            readout = purge_state_dict(state_dict=readout, purge_key=feature_key)

        if not pretrained_grid:
            readout = purge_state_dict(state_dict=readout, purge_key=grid_key)

        if not pretrained_bias:
            readout = purge_state_dict(state_dict=readout, purge_key=bias_key)


        model.load_state_dict(readout, strict=False)

    return model


def transfer_core_fullgauss_readout(dataloaders,
                                    seed,
                                    transfer_key=dict(),
                                    core_transfer_table=None,
                                    freeze_core=True,
                                    gamma_readout=4,
                                    elu_offset=0,
                                    data_info=None,
                                    readout_bias=True,
                                    init_mu_range=0.2,
                                    init_sigma=1.,
                                    gauss_type='full',
                                    grid_mean_predictor=None,
                                    share_features=False,
                                    share_grid=False,
                                    gamma_grid_dispersion=0,
                                    ):

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields[:2]

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = list(input_channels.values())[0] if isinstance(input_channels, dict) else input_channels[0]

    set_random_seed(seed)

    core_model = simple_core_transfer(dataloaders=dataloaders,
                                      seed=seed,
                                      transfer_key=transfer_key,
                                      core_transfer_table=core_transfer_table,
                                      freeze_core=freeze_core,
                                      data_info=data_info,
                                      )

    core = core_model.core

    in_shapes_dict = {
        k: get_module_output(core, v[in_name])[1:]
        for k, v in session_shape_dict.items()
    }

    readout = MultipleFullGaussian2d(
        in_shape_dict=in_shapes_dict,
        loader=dataloaders,
        n_neurons_dict=n_neurons_dict,
        init_mu_range=init_mu_range,
        bias=readout_bias,
        init_sigma=init_sigma,
        gamma_readout=gamma_readout,
        gauss_type=gauss_type,
    )

    # initializing readout bias to mean response
    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))[:2]
            readout[key].bias.data = targets.mean(0)

    model = Encoder(core=core,
                    readout=readout,
                    elu_offset=elu_offset,
                    )
    return model









