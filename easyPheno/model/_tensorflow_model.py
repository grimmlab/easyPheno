import abc
import numpy as np
import optuna
import tensorflow as tf
import joblib

from . import _base_model


class TensorflowModel(_base_model.BaseModel, abc.ABC):
    """
    Parent class based on :obj:`~easyPheno.model._base_model.BaseModel` for all TensorFlow models to share functionalities.
    See :obj:`~easyPheno.model._base_model.BaseModel` for more information.

    **Attributes**

        *Inherited attributes*

        See :obj:`~easyPheno.model._base_model.BaseModel`.

        *Additional attributes*

        - n_features (*int*): Number of input features to the model
        - width_onehot (*int*): Number of input channels in case of onehot encoding
        - batch_size (*int*): Batch size for batch-based training
        - n_epochs (*int*): Number of epochs for optimization
        - optimizer (*tf.keras.optimizers.Optimizer*): optimizer for model fitting
        - loss_fn: loss function for model fitting
        - early_stopping_patience (*int*): epochs without improvement before early stopping
        - early_stopping_point (*int*): epoch at which early stopping occured
        - early_stopping_callback (*tf.keras.callbacks.EarlyStopping*): callback for early stopping

    :param task: ML task (regression or classification) depending on target variable
    :param optuna_trial: optuna.trial.Trial : trial of optuna for optimization
    :param encoding: the encoding to use (standard encoding or user-defined)
    :param n_features: Number of input features to the model
    :param width_onehot: Number of input channels in case of onehot encoding
    :param batch_size: Batch size for batch-based training
    :param n_epochs: Number of epochs for optimization

    """

    def __init__(self, task: str, optuna_trial: optuna.trial.Trial, encoding: str = None, n_outputs: int = 1,
                 n_features: int = None, width_onehot: int = None, batch_size: int = None, n_epochs: int = None):
        self.all_hyperparams = self.common_hyperparams()  # add hyperparameters commonly optimized for all torch models
        self.n_features = n_features
        self.width_onehot = width_onehot  # relevant for models using onehot encoding e.g. CNNs
        super().__init__(task=task, optuna_trial=optuna_trial, encoding=encoding, n_outputs=n_outputs)
        self.batch_size = \
            batch_size if batch_size is not None else 2**self.suggest_hyperparam_to_optuna('batch_size_exp')
        self.n_epochs = n_epochs if n_epochs is not None else self.suggest_hyperparam_to_optuna('n_epochs')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.suggest_hyperparam_to_optuna('learning_rate'))
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) if task == 'classification' \
            else tf.keras.losses.MeanSquaredError()
        # early stopping if there is no improvement on validation loss for a certain number of epochs
        self.early_stopping_patience = self.suggest_hyperparam_to_optuna('early_stopping_patience')
        self.early_stopping_point = None
        self.early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=self.early_stopping_patience, mode='min', restore_best_weights=True
        )
        self.model.compile(self.optimizer, loss=self.loss_fn)

    def train_val_loop(self, X_train: np.array, y_train: np.array, X_val: np.array, y_val: np.array) -> np.array:
        """
        Implementation of a train and validation loop for  TensorFlow models.
        See :obj:`~easyPheno.model._base_model.BaseModel` for more information
        """
        history = self.model.fit(x=X_train, y=y_train, batch_size=self.batch_size, epochs=self.n_epochs,
                                 validation_data=(X_val, y_val), validation_freq=1, verbose=2,
                                 callbacks=[self.early_stopping_callback])
        if len(history.history['loss']) != self.n_epochs:
            self.early_stopping_point = len(history.history['loss']) - self.early_stopping_patience
            print("Early Stopping at " + str(self.early_stopping_point + self.early_stopping_patience)
                  + ' of ' + str(self.n_epochs))
        return self.predict(X_in=X_val)

    def retrain(self, X_retrain: np.array, y_retrain: np.array):
        """
        Implementation of the retraining for PyTorch models.
        See :obj:`~easyPheno.model._base_model.BaseModel` for more information
        """
        n_epochs_to_retrain = self.n_epochs if self.early_stopping_point is None else self.early_stopping_point
        self.model.fit(x=X_retrain, y=y_retrain, batch_size=self.batch_size, epochs=n_epochs_to_retrain, verbose=2)

    def predict(self, X_in: np.array) -> np.array:
        """
        Implementation of a prediction based on input features for PyTorch models.
        See :obj:`~easyPheno.model._base_model.BaseModel` for more information
        """
        dataloader = self.get_dataloader(X=X_in, shuffle=False)
        predictions = None
        for inputs in dataloader:
            outputs = self.model(inputs, training=False)
            # concat
            predictions = outputs.numpy() if predictions is None else np.vstack((predictions, outputs.numpy()))
        if self.task == 'classification':
            predictions = predictions.argmax(axis=1)
        return predictions

    def get_dataloader(self, X: np.array, y: np.array = None, shuffle: bool = True) -> tf.data.Dataset:
        """
        Get a dataloader using the specified data and batch size

        :param X: feature matrix to use
        :param y: optional target vector to use
        :param shuffle: shuffle parameter for DataLoader

        :return: batched dataset
        """
        # drop last sample if last batch would only contain one sample
        if (len(X) % self.batch_size) == 1:
            X = X[:-1]
            y = y[:-1] if y is not None else None
        dataset = tf.data.Dataset.from_tensor_slices((X, y)) if y is not None \
            else tf.data.Dataset.from_tensor_slices(X)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(dataset))
        return dataset.batch(batch_size=self.batch_size)

    @staticmethod
    def common_hyperparams():
        """
        Add hyperparameters that are common for PyTorch models.
        Do not need to be included in optimization for every child model.
        Also See :obj:`~easyPheno.model._base_model.BaseModel` for more information
        """
        return {
            'dropout': {
                'datatype': 'float',
                'lower_bound': 0,
                'upper_bound': 0.5,
                'step': 0.25
            },
            'act_function': {
                'datatype': 'categorical',
                'list_of_values': ['relu']
            },
            'batch_size_exp': {
                'datatype': 'int',
                'lower_bound': 3,
                'upper_bound': 6
            },
            'n_epochs': {
                'datatype': 'categorical',
                'list_of_values': [100, 500, 1000, 10000]
            },
            'learning_rate': {
                'datatype': 'categorical',
                'list_of_values': [1e-5, 1e-4, 1e-3, 1e-2]
            },
            'early_stopping_patience': {
                'datatype': 'int',
                'lower_bound': 0,
                'upper_bound': 10,
                'step': 10
            }
        }

    def save_model(self, path: str, filename: str):
        """
        Method to persist the whole model object on a hard drive (can be loaded with :obj:`~easyPheno.model._model_functions.load_model`)

        :param path: path where the model will be saved
        :param filename: filename of the model
        """
        optimizer = self.optimizer
        # special case for serialization of optimizer prior to saving
        self.optimizer = tf.keras.optimizers.serialize(self.optimizer)
        joblib.dump(self, path + filename, compress=3)
        self.optimizer = optimizer
