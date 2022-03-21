import abc
import numpy as np
import optuna
import tensorflow as tf
import joblib

from model import _base_model


class TensorflowModel(_base_model.BaseModel, abc.ABC):
    """
    Parent class based on BaseModel for all TensorFlow models to share functionalities
    See BaseModel for more information
    """

    def __init__(self, task: str, optuna_trial: optuna.trial.Trial, encoding: str = None, n_outputs: int = 1,
                 n_features: int = None, batch_size: int = None, n_epochs: int = None, width_onehot: int = None):
        self.all_hyperparams = self.common_hyperparams()  # add hyperparameters commonly optimized for all torch models
        self.n_features = n_features
        self.width_onehot = width_onehot
        super().__init__(task=task, optuna_trial=optuna_trial, encoding=encoding, n_outputs=n_outputs)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) if task == 'classification' \
            else tf.keras.losses.MeanSquaredError()
        self.batch_size = \
            batch_size if batch_size is not None else 2**self.suggest_hyperparam_to_optuna('batch_size_exp')
        self.n_epochs = n_epochs if n_epochs is not None else self.suggest_hyperparam_to_optuna('n_epochs')
        # optimizer to use may be included as hyperparam
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.suggest_hyperparam_to_optuna('learning_rate'))
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
        See BaseModel for more information
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
        See BaseModel for more information
        """
        n_epochs_to_retrain = self.n_epochs if self.early_stopping_point is None else self.early_stopping_point
        self.model.fit(x=X_retrain, y=y_retrain, batch_size=self.batch_size, epochs=n_epochs_to_retrain, verbose=2)

    def predict(self, X_in: np.array) -> np.array:
        """"
        Implementation of a prediction based on input features for PyTorch models.
        See BaseModel for more information
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
        Get a Pytorch DataLoader using the specified data and batch size
        :param X: feature matrix to use
        :param y: optional target vector to use
        :param shuffle: shuffle parameter for DataLoader
        :return: Pytorch DataLoader
        """
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
        Also see BaseModel for more information
        """
        return {  # TODO: ranges anpassen for start der Experimente
            'dropout': {
                'datatype': 'float',
                'lower_bound': 0,
                'upper_bound': 0.5,
                'step': 0.1
            },
            'act_function': {
                'datatype': 'categorical',
                'list_of_values': ['relu']  # , 'tanh']
            },
            'batch_size_exp': {
                'datatype': 'int',
                'lower_bound': 3,
                'upper_bound': 6
            },
            'n_epochs': {
                'datatype': 'categorical',
                'list_of_values': [50, 100, 500, 1000, 5000, 10000]  # , 50000, 100000, 500000]
            },
            'learning_rate': {
                'datatype': 'categorical',
                'list_of_values': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
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
        Method to persist the whole model object on a hard drive (can be loaded with joblib.load(filepath))
        :param path: path where the model will be saved
        :param filename: filename of the model
        """
        optimizer = self.optimizer
        self.optimizer = tf.keras.optimizers.serialize(self.optimizer)
        joblib.dump(self, path + filename, compress=3)
        self.optimizer = optimizer
