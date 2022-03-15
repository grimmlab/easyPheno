import torch

from model import _torch_model


class Cnn(_torch_model.TorchModel):
    standard_encoding = 'onehot'
    possible_encodings = ['onehot']

    def define_model(self) -> torch.nn.Sequential:
        """See BaseModel for more information"""
        padding = 0
        dilation = 1
        n_layers = self.suggest_hyperparam_to_optuna('n_layers')
        model = []
        act_function = self.get_torch_object_for_string(string_to_get=self.suggest_hyperparam_to_optuna('act_function'))
        in_channels = self.width_onehot
        kernel_size = 2 ** self.suggest_hyperparam_to_optuna('kernel_size_exp')
        stride = max(1, int(kernel_size * self.suggest_hyperparam_to_optuna('stride_perc_of_kernel_size')))
        out_channels = 2 ** self.suggest_hyperparam_to_optuna('initial_out_channels_exp')
        kernel_size_max_pool = 2 ** self.suggest_hyperparam_to_optuna('maxpool_kernel_size_exp')
        frequency_out_channels_doubling = self.suggest_hyperparam_to_optuna('frequency_out_channels_doubling')
        # Add n_layers with: Conv1d + BatchNorm + activation + Dropout
        for layer in range(n_layers):
            model.append(torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size, stride=stride))
            model.append(act_function)
            model.append(torch.nn.BatchNorm1d(num_features=out_channels))
            p = self.suggest_hyperparam_to_optuna('dropout')
            model.append(torch.nn.Dropout(p))
            model.append(torch.nn.MaxPool1d(kernel_size=kernel_size_max_pool))
            in_channels = out_channels
            if (layer+1) % frequency_out_channels_doubling:
                out_channels *= 2
        # Flatten and linear layers with dropout
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
        """See BaseModel for more information on the format"""
        return {
            'n_layers': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 3
            },
            'initial_out_channels_exp': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 3
            },
            'frequency_out_channels_doubling': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 2
            },
            'kernel_size_exp': {
                'datatype': 'int',
                'lower_bound': 3,
                'upper_bound': 6
            },
            'maxpool_kernel_size_exp': {
                'datatype': 'int',
                'lower_bound': 2,
                'upper_bound': 4
            },
            'stride_perc_of_kernel_size': {
                'datatype': 'float',
                'lower_bound': 0,
                'upper_bound': 1,
                'step': 0.1
            },
            'n_units_factor_linear_layer': {
                'datatype': 'float',
                'lower_bound': 0.1,
                'upper_bound': 1,
                'step': 0.1
            },
        }
