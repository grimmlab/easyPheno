import numpy as np
import joblib
import tensorflow as tf

from model import _base_model, _tensorflow_model


def load_retrain_model(path: str, filename: str, X_retrain: np.array, y_retrain: np.array,
                       early_stopping_point: int = None) -> _base_model.BaseModel:
    """
    Method to load and retrain persisted model

    :param path: path where the model is saved
    :param filename: filename of the model
    :param X_retrain: feature matrix for retraining
    :param y_retrain: target vector for retraining
    :param early_stopping_point: optional early stopping point relevant for some models

    :return: model instance
    """
    model = load_model(path=path, filename=filename)
    if early_stopping_point is not None:
        model.early_stopping_point = early_stopping_point
    model.retrain(X_retrain=X_retrain, y_retrain=y_retrain)
    return model


def load_model(path: str, filename: str) -> _base_model.BaseModel:
    """
    Method to load persisted model

    :param path: path where the model is saved
    :param filename: filename of the model

    :return: model instance
    """
    model = joblib.load(path + filename)
    # special case for loading tensorflow optimizer
    if issubclass(type(model), _tensorflow_model.TensorflowModel):
        model.optimizer = tf.keras.optimizers.deserialize(model.optimizer)
    return model

