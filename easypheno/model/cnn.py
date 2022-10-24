import torch

from . import _torch_model


class Cnn(_torch_model.TorchModel):
    """
    Implementation of a class for a Convolutional Neural Network (CNN).

    See :obj:`~easypheno.model._base_model.BaseModel` and :obj:`~easypheno.model._torch_model.TorchModel` for more information on the attributes.
    """
    standard_encoding = 'onehot'
    possible_encodings = ['onehot']

    def define_model(self) -> torch.nn.Sequential:
        """
        Definition of a CNN network.

        Architecture:

            - N_LAYERS of (Conv1d + BatchNorm + Dropout)
            - MaxPool1d, Flatten, Linear, BatchNorm, Dropout
            - Linear output layer

        Kernel sizes for convolutional and max pooling layers may be fixed or optimized.
        Same applies for strides, number of output channels of the first convolutional layer, dropout rate,
        frequency of a doubling of the output channels and number of units in the first linear layer.
        """
        n_layers = self.suggest_hyperparam_to_optuna('n_layers')
        model = []
        act_function = self.get_torch_object_for_string(string_to_get=self.suggest_hyperparam_to_optuna('act_function'))
        in_channels = self.width_onehot
        kernel_size = self.suggest_hyperparam_to_optuna('kernel_size')
        stride = max(1, int(kernel_size * self.suggest_hyperparam_to_optuna('stride_perc_of_kernel_size')))
        out_channels = 2 ** 2  # self.suggest_hyperparam_to_optuna('initial_out_channels_exp')
        frequency_out_channels_doubling = 2  # self.suggest_hyperparam_to_optuna('frequency_out_channels_doubling')
        p = self.suggest_hyperparam_to_optuna('dropout')
        for layer in range(n_layers):
            model.append(torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size, stride=stride))
            model.append(act_function)
            model.append(torch.nn.BatchNorm1d(num_features=out_channels))
            model.append(torch.nn.Dropout(p))
            in_channels = out_channels
            if ((layer+1) % frequency_out_channels_doubling) == 0:
                out_channels *= 2
        model.append(torch.nn.MaxPool1d(kernel_size=kernel_size))
        model.append(torch.nn.Flatten())
        in_features = torch.nn.Sequential(*model)(torch.zeros(size=(1, self.width_onehot, self.n_features))).shape[1]
        out_features = int(in_features * self.suggest_hyperparam_to_optuna('n_units_factor_linear_layer'))
        model.append(torch.nn.Linear(in_features=in_features, out_features=out_features))
        model.append(act_function)
        model.append(torch.nn.BatchNorm1d(num_features=out_features))
        model.append(torch.nn.Dropout(p))
        model.append(torch.nn.Linear(in_features=out_features, out_features=self.n_outputs))
        return torch.nn.Sequential(*model)

    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~easypheno.model._base_model.BaseModel` for more information on the format.

        See :obj:`~easypheno.model._torch_model.TorchModel` for more information on hyperparameters common for all torch models.
        """
        kernel_size = {
            'datatype': 'int',
            'lower_bound': 2,
            'upper_bound': 8
        }
        n_units_factor_linear_layer = {
            # Number of units in the linear layer after flattening in relation to the number of inputs
            'datatype': 'float',
            'lower_bound': 0.2,
            'upper_bound': 1,
            'step': 0.05
        }

        if self.n_features > 15000:
            kernel_size = {
                'datatype': 'int',
                'lower_bound': 4,
                'upper_bound': 10,
                'step': 2
            }
            n_units_factor_linear_layer = {
                # Number of units in the linear layer after flattening in relation to the number of inputs
                'datatype': 'float',
                'lower_bound': 0.2,
                'upper_bound': 0.5,
                'step': 0.05
            }
        if self.n_features > 50000:
            kernel_size = {
                'datatype': 'int',
                'lower_bound': 8,
                'upper_bound': 14,
                'step': 2
            }
        if self.n_features > 100000:
            kernel_size = {
                'datatype': 'int',
                'lower_bound': 10,
                'upper_bound': 25,
                'step': 5
            }
        return {
            'n_layers': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 3
            },
            'initial_out_channels_exp': {
                # Exponent with base 2 to get number of output channels of the first convolutional layer
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 2
            },
            'frequency_out_channels_doubling': {
                # Frequency of doubling the initial output channels after a certain number of layers
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 2
            },
            'kernel_size': kernel_size,
            'stride_perc_of_kernel_size': {
                # Stride in relation to the kernel size
                'datatype': 'float',
                'lower_bound': 0.5,
                'upper_bound': 1,
                'step': 0.1
            },
            'n_units_factor_linear_layer': n_units_factor_linear_layer
        }
