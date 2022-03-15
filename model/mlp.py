import torch

from model import _torch_model


class Mlp(_torch_model.TorchModel):
    standard_encoding = '012'
    possible_encodings = ['012']

    def define_model(self) -> torch.nn.Sequential:
        """See BaseModel for more information"""
        n_layers = self.suggest_hyperparam_to_optuna('n_layers')
        model = []
        act_function = self.get_torch_object_for_string(string_to_get=self.suggest_hyperparam_to_optuna('act_function'))
        in_features = self.n_features
        out_features = int(in_features * self.suggest_hyperparam_to_optuna('n_initial_units_factor'))
        for layer in range(n_layers):
            model.append(torch.nn.Linear(in_features=in_features, out_features=out_features))
            model.append(act_function)
            model.append(torch.nn.BatchNorm1d(num_features=out_features))
            p = self.suggest_hyperparam_to_optuna('dropout')
            model.append(torch.nn.Dropout(p=p))
            in_features = out_features
            out_features = int(in_features * (1-self.suggest_hyperparam_to_optuna('perc_decrease_per_layer')))
        model.append(torch.nn.Linear(in_features=in_features, out_features=self.n_outputs))
        return torch.nn.Sequential(*model)

    def define_hyperparams_to_tune(self) -> dict:
        """See BaseModel for more information on the format"""
        return {  # TODO: ranges anpassen for start der Experimente
            'n_layers': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 3
            },
            'n_initial_units_factor': {
                'datatype': 'float',
                'lower_bound': 0.5,
                'upper_bound': 1,
                'step': 0.1
            },
            'perc_decrease_per_layer': {
                'datatype': 'float',
                'lower_bound': 0.1,
                'upper_bound': 0.5,
                'step': 0.1
            }
        }
