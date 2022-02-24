from model import base_model
import abc
import numpy as np


class SklearnModel(base_model.BaseModel, abc.ABC):
    """
    Parent class based on BaseModel for all models with a sklearn-like API to share functionalities
    See BaseModel for more information
    """

    def retrain(self, X_retrain: np.array, y_retrain: np.array):
        """
        Implementation of the retraining for models with sklearn-like API.
        See BaseModel for more information
        """
        self.model.fit(X_retrain, y_retrain)

    def predict(self, X_in: np.array) -> np.array:
        """
        Implementation of a prediction based on input features for models with sklearn-like API.
        See BaseModel for more information
        """
        return np.reshape(self.model.predict(X_in), (-1, 1))

    def train_val_loop(self, X_train: np.array, y_train: np.array, X_val: np.array, y_val: np.array) -> np.array:
        """
        Implementation of a train and validation loop for models with sklearn-like API.
        See BaseModel for more information
        """
        # train model
        self.model.fit(X_train, y_train)
        # validate model
        return self.predict(X_in=X_val)
