import torch

from model import torch_model


class Mlp(torch_model.TorchModel):
    standard_encoding = '012'
    possible_encodings = ['012', 'raw', 'onehot']

    def define_model(self) -> torch.nn.Sequential:
        """See BaseModel for more information"""
        n_layers = self.suggest_hyperparam_to_optuna('n_layers')
        model = []
        act_function = self.suggest_hyperparam_to_optuna('act_function')
        in_features = self.n_features
        for layer in range(n_layers):
            out_features = self.suggest_hyperparam_to_optuna('n_units_per_layer')
            model.append(torch.nn.Linear(in_features=in_features, out_features=out_features))
            model.append(act_function)

            in_features = out_features
        return None

    def define_hyperparams_to_tune(self) -> dict:
        """See BaseModel for more information on the format"""
        return {
            'n_units_per_layer': {
                'datatype': 'int',
                'lower_bound': 2**2,
                'upper_bound': 2**16,
                'log': True
            },

        }
