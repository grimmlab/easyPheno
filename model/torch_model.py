import base_model
import abc
import numpy as np


class TorchModel(base_model.BaseModel, abc.ABC):
    """
    Parent class based on BaseModel for all PyTorch models to share functionalities
    See BaseModel for more information
    """
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
