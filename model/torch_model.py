import abc
import numpy as np
import optuna
import torch.nn

from model import base_model
from preprocess import base_dataset


class TorchModel(base_model.BaseModel, abc.ABC):
    """
    Parent class based on BaseModel for all PyTorch models to share functionalities
    See BaseModel for more information
    """

    def __init__(self, task: str, optuna_trial: optuna.trial.Trial, encoding: str = None,
                 n_features: int = None):
        super().__init__(task=task, optuna_trial=optuna_trial, encoding=encoding)
        self.add_common_hyperparams()  # add hyperparameters that are commonly optimized for all torch models
        self.n_features = n_features

    # epochen + batch-based training einbauen
    def train(self, X_train: np.array, y_train: np.array):
        """
        Implementation of one train iteration for PyTorch models.
        See BaseModel for more information
        """
        pass

    def predict(self, X_in: np.array) -> np.array:
        """"
        Implementation of a prediction based on input features for PyTorch models.
        See BaseModel for more information
        """
        pass

    def add_common_hyperparams(self):
        """
        Add hyperparameters that are common for PyTorch models.
        Do not need to be included in optimization for every child model.
        Also see BaseModel for more information
        """
        common_hyperparams_to_add = {
            'n_layers': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 10
            },
            'dropout_proba': {
                'datatype': 'float',
                'lower_bound': 0,
                'upper_bound': 1
            },
            'act_function': {
                'datatype': 'categorical',
                'list_of_values': [torch.nn.ReLU(), torch.nn.Tanh()]  # TODO: testen
            }
        }
        self.all_hyperparams.update(common_hyperparams_to_add)
