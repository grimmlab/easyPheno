import torch

from model import torch_model


class Mlp(torch_model.TorchModel):
    standard_encoding = '012'
    possible_encodings = ['012', 'raw']

    def define_model(self) -> torch.nn.Sequential:
        """See BaseModel for more information"""
        n_layers = self.suggest_hyperparam_to_optuna('n_layers')
        model = []
        act_function = self.get_torch_object_for_string(string_to_get=self.suggest_hyperparam_to_optuna('act_function'))
        in_features = self.n_features
        for layer in range(n_layers):
            out_features = 2 ** self.suggest_hyperparam_to_optuna('n_units_per_layer_exp')
            model.append(torch.nn.Linear(in_features=in_features, out_features=out_features))
            model.append(torch.nn.BatchNorm1d(num_features=out_features))
            model.append(act_function)
            p = self.suggest_hyperparam_to_optuna('dropout')
            model.append(torch.nn.Dropout(p=p))
            in_features = out_features
        model.append(torch.nn.Linear(in_features=in_features, out_features=self.n_outputs))
        return torch.nn.Sequential(*model)

    def define_hyperparams_to_tune(self) -> dict:
        """See BaseModel for more information on the format"""
        return {
            'n_layers': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 4  # 10
            },
            'n_units_per_layer_exp': {
                'datatype': 'int',
                'lower_bound': 2,
                'upper_bound': 6  # 10
            },
        }
