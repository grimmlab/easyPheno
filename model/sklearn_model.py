import base_model
import abc
import numpy as np
import sklearn


class SklearnModel(base_model.BaseModel, abc.ABC):
    """
    Parent class based on BaseModel for all models with a sklearn-like API to share functionalities
    """

    def train(self, X_train: np.array, y_train: np.array):
        """See BaseModel for more information"""
        self.model.fit(X_train, y_train)

    def predict(self, X_in: np.array) -> np.array:
        """See BaseModel for more information"""
        return self.model.predict(X_in)

    def reset_model(self):
        """See BaseModel for more information"""
        self.model = sklearn.base.clone(self.model)
