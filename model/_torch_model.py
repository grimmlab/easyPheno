import abc
import numpy as np
import optuna
import torch.nn
import torch.utils.data
import copy

from model import _base_model


class TorchModel(_base_model.BaseModel, abc.ABC):
    """
    Parent class based on BaseModel for all PyTorch models to share functionalities
    See BaseModel for more information
    """

    def __init__(self, task: str, optuna_trial: optuna.trial.Trial, encoding: str = None, n_outputs: int = 1,
                 n_features: int = None, batch_size: int = None, n_epochs: int = None, width_onehot: int = None):
        self.all_hyperparams = self.common_hyperparams()  # add hyperparameters commonly optimized for all torch models
        self.n_features = n_features
        self.width_onehot = width_onehot
        super().__init__(task=task, optuna_trial=optuna_trial, encoding=encoding, n_outputs=n_outputs)
        self.loss_fn = torch.nn.CrossEntropyLoss() if task == 'classification' else torch.nn.MSELoss()
        self.batch_size = \
            batch_size if batch_size is not None else 2**self.suggest_hyperparam_to_optuna('batch_size_exp')
        self.n_epochs = n_epochs if n_epochs is not None else self.suggest_hyperparam_to_optuna('n_epochs')
        # optimizer to use may be included as hyperparam
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=self.suggest_hyperparam_to_optuna('learning_rate'))
        # early stopping if there is no improvement on validation loss for a certain number of epochs
        self.early_stopping_patience = self.suggest_hyperparam_to_optuna('early_stopping_patience')
        self.early_stopping_point = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def train_val_loop(self, X_train: np.array, y_train: np.array, X_val: np.array, y_val: np.array) -> np.array:
        """
        Implementation of a train and validation loop for  PyTorch models.
        See BaseModel for more information
        """
        train_loader = self.get_dataloader(X=X_train, y=y_train)
        # del X_train, y_train
        val_loader = self.get_dataloader(X=X_val, y=y_val)
        # del y_val
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
            if epochs_wo_improvement >= self.early_stopping_patience:
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
        See BaseModel for more information
        """
        retrain_loader = self.get_dataloader(X=X_retrain, y=y_retrain)
        # del X_retrain, y_retrain
        n_epochs_to_retrain = self.n_epochs if self.early_stopping_point is None else self.early_stopping_point
        self.model.to(device=self.device)
        for epoch in range(n_epochs_to_retrain):
            print('Retrain: Epoch ' + str(epoch + 1) + ' of ' + str(n_epochs_to_retrain))
            self.train_one_epoch(retrain_loader)

    def predict(self, X_in: np.array) -> np.array:
        """"
        Implementation of a prediction based on input features for PyTorch models.
        See BaseModel for more information
        """
        dataloader = self.get_dataloader(X=X_in, shuffle=False)
        # del X_in
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
        if (len(X) % self.batch_size) == 1:
            X = X[:-1]
            y = y[:-1] if y is not None else None
        X = torch.from_numpy(X).float()
        if self.encoding == 'onehot':
            X = torch.swapaxes(X, 1, 2)
        y = torch.reshape(torch.from_numpy(y).float(), (-1, 1)) if y is not None else None
        y = y.flatten() if (self.task == 'classification' and y is not None) else y
        dataset = torch.utils.data.TensorDataset(X, y) if y is not None \
            else X
        # del X, y
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=shuffle)

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

    @staticmethod
    def get_torch_object_for_string(string_to_get: str):
        string_to_object_dict = {
            'relu': torch.nn.ReLU(),
            'tanh': torch.nn.Tanh()
        }
        return string_to_object_dict[string_to_get]
