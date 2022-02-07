from model import base_model
import abc
import numpy as np


class SklearnModel(base_model.BaseModel, abc.ABC):
    """
    Parent class based on BaseModel for all models with a sklearn-like API to share functionalities
    See BaseModel for more information
    """

    def train(self, X_train: np.array, y_train: np.array):
        """
        Implementation of one train iteration for models with sklearn-like API.
        See BaseModel for more information
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_in: np.array) -> np.array:
        """
        Implementation of a prediction based on input features for models with sklearn-like API.
        See BaseModel for more information
        """
        return self.model.predict(X_in)
