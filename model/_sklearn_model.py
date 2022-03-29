from model import _base_model
import abc
import numpy as np


class SklearnModel(_base_model.BaseModel, abc.ABC):
    """
    Parent class based on :obj:`~model._base_model.BaseModel` for all models with a sklearn-like API to share
    functionalities. See :obj:`~model._base_model.BaseModel` for more information.

    ## Attributes ##
        # Inherited attributes #

        See :obj:`~model._base_model.BaseModel`
    """

    def retrain(self, X_retrain: np.array, y_retrain: np.array):
        """
        Implementation of the retraining for models with sklearn-like API.
        See :obj:`~model._base_model.BaseModel` for more information.
        """
        self.model.fit(X_retrain, np.ravel(y_retrain))

    def predict(self, X_in: np.array) -> np.array:
        """
        Implementation of a prediction based on input features for models with sklearn-like API.
        See :obj:`~model._base_model.BaseModel` for more information.
        """
        return np.reshape(self.model.predict(X_in), (-1, 1))

    def train_val_loop(self, X_train: np.array, y_train: np.array, X_val: np.array, y_val: np.array) -> np.array:
        """
        Implementation of a train and validation loop for models with sklearn-like API.
        See :obj:`~model._base_model.BaseModel` for more information.
        """
        self.model.fit(X_train, np.ravel(y_train))
        return self.predict(X_in=X_val)
