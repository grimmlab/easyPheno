import numpy as np

from . import _param_free_base_model


class Blup(_param_free_base_model.ParamFreeBaseModel):
    """
    Implementation of a class for BLUP.

    See :obj:`~easyPheno.model._param_free_base_model.ParamFreeBaseModel` for more information on the attributes.
    """
    standard_encoding = '012'
    possible_encodings = ['012']

    def define_model(self):
        """
        Implementation of a BLUP prediction model.
        See :obj:`~easyPheno.model._param_free_base_model.ParamFreeBaseModel` for more information.
        """
        return 'Ente'

    def fit(self, X: np.array, y: np.array) -> np.array:
        """
        Implementation of fit function for BLUP.
        See :obj:`~easyPheno.model._param_free_base_model.ParamFreeBaseModel` for more information.
        """
        return y + 0.2

    def predict(self, X_in: np.array) -> np.array:
        """
        Implementation of predict function for BLUP.
        See :obj:`~easyPheno.model._param_free_base_model.ParamFreeBaseModel` for more information.
        """
        return np.ones(shape=(X_in.shape[0], 1))
