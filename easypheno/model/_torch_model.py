import abc
import numpy as np
import optuna
import torch.nn
import torch.utils.data
import copy

from . import _base_model


class TorchModel(_base_model.BaseModel, abc.ABC):
    """
    Parent class based on :obj:`~easypheno.model._base_model.BaseModel` for all PyTorch models to share functionalities.
    See :obj:`~easypheno.model._base_model.BaseModel` for more information.

    *Attributes*

        *Inherited attributes*

        See :obj:`~easypheno.model._base_model.BaseModel`.

        *Additional attributes*

        - n_features (*int*): Number of input features to the model
        - width_onehot (*int*): Number of input channels in case of onehot encoding
        - batch_size (*int*): Batch size for batch-based training
        - n_epochs (*int*): Number of epochs for optimization
        - optimizer (*torch.optim.optimizer.Optimizer*): optimizer for model fitting
        - loss_fn: loss function for model fitting
        - early_stopping_patience (*int*): epochs without improvement before early stopping
        - early_stopping_point (*int*): epoch at which early stopping occured
        - device (*torch.device*): device to use, e.g. GPU

    :param task: ML task (regression or classification) depending on target variable
    :param optuna_trial: optuna.trial.Trial : trial of optuna for optimization
    :param encoding: the encoding to use (standard encoding or user-defined)
    :param n_outputs: Number of outputs of the model
    :param n_features: Number of input features to the model
    :param width_onehot: Number of input channels in case of onehot encoding
    :param batch_size: Batch size for batch-based training
    :param n_epochs: Number of epochs for optimization
    :param early_stopping_point: Stop training at defined epoch
    """

    def __init__(self, task: str, optuna_trial: optuna.trial.Trial, encoding: str = None, n_outputs: int = 1,
                 n_features: int = None, width_onehot: int = None, batch_size: int = None, n_epochs: int = None,
                 early_stopping_point: int = None):
        self.all_hyperparams = self.common_hyperparams()  # add hyperparameters commonly optimized for all torch models
        self.n_features = n_features
        self.width_onehot = width_onehot  # relevant for models using onehot encoding e.g. CNNs
        super().__init__(task=task, optuna_trial=optuna_trial, encoding=encoding, n_outputs=n_outputs)
        self.batch_size = \
            batch_size if batch_size is not None else self.suggest_hyperparam_to_optuna('batch_size')
        self.n_epochs = n_epochs if n_epochs is not None else self.suggest_hyperparam_to_optuna('n_epochs')
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=self.suggest_hyperparam_to_optuna('learning_rate'))
        self.loss_fn = torch.nn.CrossEntropyLoss() if task == 'classification' else torch.nn.MSELoss()
        # self.l1_factor = self.suggest_hyperparam_to_optuna('l1_factor')
        # early stopping if there is no improvement on validation loss for a certain number of epochs
        self.early_stopping_patience = self.suggest_hyperparam_to_optuna('early_stopping_patience')
        self.early_stopping_point = early_stopping_point
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def train_val_loop(self, X_train: np.array, y_train: np.array, X_val: np.array, y_val: np.array) -> np.array:
        """
        Implementation of a train and validation loop for  PyTorch models.
        See :obj:`~easypheno.model._base_model.BaseModel` for more information
        """
        train_loader = self.get_dataloader(X=X_train, y=y_train)
        val_loader = self.get_dataloader(X=X_val, y=y_val)
        best_model = copy.deepcopy(self.model)
        self.model.to(device=self.device)
        best_loss = None
        epochs_wo_improvement = 0
        for epoch in range(self.n_epochs):
            self.train_one_epoch(train_loader=train_loader)
            val_loss = self.validate_one_epoch(val_loader=val_loader)
            if best_loss is None or val_loss < best_loss:
                best_loss = val_loss
                epochs_wo_improvement = 0
                best_model = copy.deepcopy(self.model)
            else:
                epochs_wo_improvement += 1
            print('Epoch ' + str(epoch + 1) + ' of ' + str(self.n_epochs))
            print('Current val_loss=' + str(val_loss) + ', best val_loss=' + str(best_loss))
            if epoch >= 20 and epochs_wo_improvement >= self.early_stopping_patience:
                print("Early Stopping at " + str(epoch + 1) + ' of ' + str(self.n_epochs))
                self.early_stopping_point = epoch - self.early_stopping_patience
                self.model = best_model
                return self.predict(X_in=X_val)
        return self.predict(X_in=X_val)

    def train_one_epoch(self, train_loader: torch.utils.data.DataLoader):
        """
        Train one epoch

        :param train_loader: DataLoader with training data
        """
        self.model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device=self.device), targets.to(device=self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.get_loss(outputs=outputs, targets=targets)
            # l1_loss = 0
            # for param in self.model.parameters():
            #     l1_loss += torch.sum(torch.abs(param))
            # loss += self.l1_factor * l1_loss
            loss.backward()
            self.optimizer.step()

    def validate_one_epoch(self, val_loader: torch.utils.data.DataLoader) -> float:
        """
        Validate one epoch

        :param val_loader: DataLoader with validation data

        :return: loss based on loss-criterion
        """
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device=self.device), targets.to(device=self.device)
                outputs = self.model(inputs)
                total_loss += self.get_loss(outputs=outputs, targets=targets).item()
        return total_loss / len(val_loader.dataset)

    def retrain(self, X_retrain: np.array, y_retrain: np.array):
        """
        Implementation of the retraining for PyTorch models.
        See :obj:`~easypheno.model._base_model.BaseModel` for more information
        """
        retrain_loader = self.get_dataloader(X=X_retrain, y=y_retrain)
        n_epochs_to_retrain = self.n_epochs if self.early_stopping_point is None else self.early_stopping_point
        self.model.to(device=self.device)
        for epoch in range(n_epochs_to_retrain):
            print('Retrain: Epoch ' + str(epoch + 1) + ' of ' + str(n_epochs_to_retrain))
            self.train_one_epoch(retrain_loader)

    def predict(self, X_in: np.array) -> np.array:
        """
        Implementation of a prediction based on input features for PyTorch models.
        See :obj:`~easypheno.model._base_model.BaseModel` for more information
        """
        dataloader = self.get_dataloader(X=X_in, shuffle=False)
        self.model.eval()
        predictions = None
        with torch.no_grad():
            for inputs in dataloader:
                inputs = inputs.to(device=self.device)
                outputs = self.model(inputs)
                predictions = torch.clone(outputs) if predictions is None else torch.cat((predictions, outputs))
        if self.task == 'classification':
            _, predictions = torch.max(predictions, 1)
        return predictions.cpu().detach().numpy()

    def get_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss based on the outputs and targets

        :param outputs: outputs of the model
        :param targets: targets of the dataset

        :return: loss
        """
        if type(self.loss_fn) in [torch.nn.CrossEntropyLoss, torch.nn.NLLLoss]:
            targets = targets.long()
        return self.loss_fn(outputs, targets)

    def get_dataloader(self, X: np.array, y: np.array = None, shuffle: bool = True) -> torch.utils.data.DataLoader:
        """
        Get a Pytorch DataLoader using the specified data and batch size

        :param X: feature matrix to use
        :param y: optional target vector to use
        :param shuffle: shuffle parameter for DataLoader

        :return: Pytorch DataLoader
        """
        # drop last sample if last batch would only contain one sample
        if (len(X) % self.batch_size) == 1:
            X = X[:-1]
            y = y[:-1] if y is not None else None
        X = torch.from_numpy(X).float()
        if self.encoding == 'onehot':
            # Adapt to PyTorch ordering (BATCH_SIZE, CHANNELS, SIGNAL)
            X = torch.swapaxes(X, 1, 2)
        y = torch.reshape(torch.from_numpy(y).float(), (-1, 1)) if y is not None else None
        y = torch.flatten(y) if (self.task == 'classification' and y is not None) else y
        dataset = torch.utils.data.TensorDataset(X, y) if y is not None \
            else X
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=shuffle)

    @staticmethod
    def common_hyperparams():
        """
        Add hyperparameters that are common for PyTorch models.
        Do not need to be included in optimization for every child model.
        Also See :obj:`~easypheno.model._base_model.BaseModel` for more information
        """
        return {
            'dropout': {
                'datatype': 'float',
                'lower_bound': 0,
                'upper_bound': 0.5,
                'step': 0.1
            },
            'act_function': {
                'datatype': 'categorical',
                'list_of_values': ['relu', 'tanh']
            },
            'batch_size': {
                'datatype': 'categorical',
                'list_of_values': [4, 8, 16, 32, 64]
            },
            'n_epochs': {
                'datatype': 'categorical',
                'list_of_values': [100, 500, 1000, 5000, 10000]
            },
            'learning_rate': {
                'datatype': 'categorical',
                'list_of_values': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
            },
            'early_stopping_patience': {
                'datatype': 'int',
                'lower_bound': 0,
                'upper_bound': 20,
                'step': 5
            },
            'l1_factor': {
                'datatype': 'float',
                'lower_bound': 0,
                'upper_bound': 10**3
            }
        }

    @staticmethod
    def get_torch_object_for_string(string_to_get: str):
        """
        Get the torch object for a specific string, e.g. when suggesting to optuna as hyperparameter

        :param string_to_get: string to retrieve the torch object

        :return: torch object
        """
        string_to_object_dict = {
            'relu': torch.nn.ReLU(),
            'tanh': torch.nn.Tanh()
        }
        return string_to_object_dict[string_to_get]
