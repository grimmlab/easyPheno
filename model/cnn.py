import torch

from model import torch_model


class Cnn(torch_model.TorchModel):
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
        width = self.n_features
        for layer in range(n_layers):
            out_channels = 2 ** self.suggest_hyperparam_to_optuna('out_channels_exp')
            kernel_size = 2 ** self.suggest_hyperparam_to_optuna('kernel_size_exp')
            stride = 1
            model.append(torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size, stride=1))
            model.append(torch.nn.BatchNorm1d(num_features=out_channels))
            model.append(act_function)
            p = self.suggest_hyperparam_to_optuna('dropout')
            model.append(torch.nn.Dropout(p))
            in_channels = out_channels
            width = int((width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
        kernel_size_max_pool = 2 ** self.suggest_hyperparam_to_optuna('kernel_size_exp')
        model.append(torch.nn.MaxPool1d(kernel_size=kernel_size_max_pool))
        stride = kernel_size_max_pool
        n_out_max_pool = int((width + 2*padding - dilation * (kernel_size_max_pool - 1) - 1) / stride + 1)
        model.append(torch.nn.Flatten())
        out_features = 2 ** self.suggest_hyperparam_to_optuna('n_units_per_layer_exp')
        model.append(torch.nn.Linear(in_features=n_out_max_pool * out_channels, out_features=out_features))
        model.append(act_function)
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
                'upper_bound': 4  # 10
            },
            'out_channels_exp': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 6
            },
            'kernel_size_exp': {
                'datatype': 'int',
                'lower_bound': 2,
                'upper_bound': 6  # 8
            },
            'n_units_per_layer_exp': {
                'datatype': 'int',
                'lower_bound': 2,
                'upper_bound': 6  # 10
            }
        }
