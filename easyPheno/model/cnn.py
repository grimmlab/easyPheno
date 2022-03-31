import torch

from . import _torch_model


class Cnn(_torch_model.TorchModel):
    """
    Implementation of a class for a Convolutional Neural Network (CNN).

    See :obj:`~model._base_model.BaseModel` and :obj:`~model._torch_model.TorchModel` for more information on the attributes.
    """
    standard_encoding = 'onehot'
    possible_encodings = ['onehot']

    def define_model(self) -> torch.nn.Sequential:
        """
        Definition of a CNN network.

        Architecture:

            - N_LAYERS of (Conv1d + BatchNorm + Dropout + MaxPool1d)
            - Flatten, Linear, BatchNorm, Dropout
            - Linear output layer

        Kernel sizes for convolutional and max pooling layers may be fixed or optimized.
        Same applies for strides, number of output channels of the first convolutional layer,
        frequency of a doubling of the output channels and number of units in the first linear layer.
        """
        n_layers = self.suggest_hyperparam_to_optuna('n_layers')
        model = []
        act_function = self.get_torch_object_for_string(string_to_get=self.suggest_hyperparam_to_optuna('act_function'))
        in_channels = self.width_onehot
        factor_for_kernels = 2 if self.n_features >= 10000 else 1
        kernel_size = factor_for_kernels * self.suggest_hyperparam_to_optuna('kernel_size_factor')
        stride = max(1, int(kernel_size * self.suggest_hyperparam_to_optuna('stride_perc_of_kernel_size')))
        out_channels = 2 ** self.suggest_hyperparam_to_optuna('initial_out_channels_exp')
        kernel_size_max_pool = factor_for_kernels * self.suggest_hyperparam_to_optuna('maxpool_kernel_size_factor')
        frequency_out_channels_doubling = 2  # self.suggest_hyperparam_to_optuna('frequency_out_channels_doubling')
        for layer in range(n_layers):
            model.append(torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size, stride=stride))
            model.append(act_function)
            model.append(torch.nn.BatchNorm1d(num_features=out_channels))
            p = self.suggest_hyperparam_to_optuna('dropout')
            model.append(torch.nn.Dropout(p))
            in_channels = out_channels
            if ((layer+1) % frequency_out_channels_doubling) == 0:
                out_channels *= 2
        model.append(torch.nn.MaxPool1d(kernel_size=kernel_size_max_pool))
        model.append(torch.nn.Flatten())
        in_features = torch.nn.Sequential(*model)(torch.zeros(size=(1, self.width_onehot, self.n_features))).shape[1]
        out_features = int(in_features * self.suggest_hyperparam_to_optuna('n_units_factor_linear_layer'))
        model.append(torch.nn.Linear(in_features=in_features, out_features=out_features))
        model.append(act_function)
        model.append(torch.nn.BatchNorm1d(num_features=out_features))
        p = self.suggest_hyperparam_to_optuna('dropout')
        model.append(torch.nn.Dropout(p))
        model.append(torch.nn.Linear(in_features=out_features, out_features=self.n_outputs))
        return torch.nn.Sequential(*model)

    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~model._base_model.BaseModel` for more information on the format.

        See :obj:`~model._torch_model.TorchModel` for more information on hyperparameters common for all torch models.
        """
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
            'kernel_size_factor': {
                # Value directly used as kernel_size for the convolutional layers for self.n_features < 10.000
                # Multiplied with factor 2 for bigger values
                'datatype': 'int',
                'lower_bound': 3,
                'upper_bound': 5
            },
            'maxpool_kernel_size_factor': {
                # Value directly used as kernel_size for the maxpool layer for self.n_features < 10.000
                # Multiplied with factor 2 for bigger values
                'datatype': 'int',
                'lower_bound': 3,
                'upper_bound': 5
            },
            'stride_perc_of_kernel_size': {
                # Stride in relation to the kernel size
                'datatype': 'float',
                'lower_bound': 0,
                'upper_bound': 1,
                'step': 0.5
            },
            'n_units_factor_linear_layer': {
                # Number of units in the linear layer after flattening in relation to the number of inputs
                'datatype': 'float',
                'lower_bound': 0.2,
                'upper_bound': 0.8,
                'step': 0.2
            },
        }
