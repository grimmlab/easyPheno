import abc
import joblib
import numpy as np
import pathlib


class ParamFreeBaseModel(abc.ABC):
    """
    BaseModel parent class for all models that do not have hyperparameters, e.g. BLUP.

    Every model must be based on :obj:`~easypheno.model.param_free_base_model.ParamFreeBaseModel` directly or ParamFreeBaseModel's child classes.

    Please add ``super().__init__(PARAMS)`` to the constructor in case you override it in a child class

    **Attributes**

        *Class attributes*

        - standard_encoding (*str*): the standard encoding for this model
        - possible_encodings (*List<str>*): a list of all encodings that are possible according to the model definition

        *Instance attributes*

        - task (*str*): ML task ('regression' or 'classification') depending on target variable
        - encoding (*str*): the encoding to use (standard encoding or user-defined)


    :param task: ML task (regression or classification) depending on target variable
    :param encoding: the encoding to use (standard encoding or user-defined)

    """

    # Class attributes #
    @property
    @classmethod
    @abc.abstractmethod
    def standard_encoding(cls):
        """the standard encoding for this model"""
        raise NotImplementedError

    @property
    @classmethod
    @abc.abstractmethod
    def possible_encodings(cls):
        """a list of all encodings that are possible according to the model definition"""
        raise NotImplementedError

    # Constructor super class #
    def __init__(self, task: str, encoding: str = None):
        self.task = task
        self.encoding = self.standard_encoding if encoding is None else encoding

    # Methods required by each child class #

    @abc.abstractmethod
    def fit(self, X: np.array, y: np.array) -> np.array:
        """
        Method that fits the model based on features X and targets y

        :param X: feature matrix for retraining
        :param y: target vector

        :return: numpy array with values predicted for X
        """

    @abc.abstractmethod
    def predict(self, X_in: np.array) -> np.array:
        """
        Method that predicts target values based on the input X_in

        :param X_in: feature matrix as input

        :return: numpy array with the predicted values
        """

    def save_model(self, path: pathlib.Path, filename: str):
        """
        Persist the whole model object on a hard drive
        (can be loaded with :obj:`~easypheno.model._model_functions.load_model`)

        :param path: path where the model will be saved
        :param filename: filename of the model
        """
        joblib.dump(self, path.joinpath(filename), compress=3)
