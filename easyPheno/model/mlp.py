import torch

from . import _torch_model


class Mlp(_torch_model.TorchModel):
    """
    Implementation of a class for a feedforward Multilayer Perceptron (MLP).

    See :obj:`~easyPheno.model._base_model.BaseModel` and :obj:`~easyPheno.model._torch_model.TorchModel` for more information on the attributes.
    """
    standard_encoding = '012'
    possible_encodings = ['012']

    def define_model(self) -> torch.nn.Sequential:
        """
        Definition of an MLP network.

        Architecture:

            - N_LAYERS of (Linear + BatchNorm + Dropout)
            - Linear output layer

        Number of units in the first linear layer and percentage decrease after each may be fixed or optimized.
        """
        n_layers = self.suggest_hyperparam_to_optuna('n_layers')
        model = []
        act_function = self.get_torch_object_for_string(string_to_get=self.suggest_hyperparam_to_optuna('act_function'))
        in_features = self.n_features
        out_features = int(in_features * self.suggest_hyperparam_to_optuna('n_initial_units_factor'))
        p = self.suggest_hyperparam_to_optuna('dropout')
        perc_decrease = self.suggest_hyperparam_to_optuna('perc_decrease_per_layer')
        for layer in range(n_layers):
            model.append(torch.nn.Linear(in_features=in_features, out_features=out_features))
            model.append(act_function)
            model.append(torch.nn.BatchNorm1d(num_features=out_features))
            model.append(torch.nn.Dropout(p=p))
            in_features = out_features
            out_features = int(in_features * (1-perc_decrease))
        model.append(torch.nn.Linear(in_features=in_features, out_features=self.n_outputs))
        return torch.nn.Sequential(*model)

    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~easyPheno.model._base_model.BaseModel` for more information on the format.

        See :obj:`~easyPheno.model._torch_model.TorchModel` for more information on hyperparameters common for all torch models.
        """
        return {
            'n_layers': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 3
            },
            'n_initial_units_factor': {
                # Number of units in the first linear layer in relation to the number of inputs
                'datatype': 'float',
                'lower_bound': 0.02,
                'upper_bound': 0.1,
                'step': 0.02
            },
            'perc_decrease_per_layer': {
                # Percentage decrease of the number of units per layer
                'datatype': 'float',
                'lower_bound': 0.2,
                'upper_bound': 0.5,
                'step': 0.05
            }
        }
