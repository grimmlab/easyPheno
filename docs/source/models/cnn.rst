Convolutional Neural Network
=============================================
Subsequently, we give details on our implementation of a Convolutional Neural Network (CNN).
References for a more detailed theoretical background can be found at the end of this page, which were also used for writing this text.
We use PyTorch for our implementation. For more information on specific PyTorch objects that we use,
e.g. layers, see the `PyTorch documentation <https://pytorch.org/docs/stable/index.html>`_.

For CNN, we one-hot encoded the data, as this data can be easily processed by a CNN.
This type of encoding preserves the whole nucleotide information and might thus lead to a smaller information loss than other encodings.

Some of the methods and attributes relevant for the CNN are already defined in its parent class `TorchModel <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/_torch_model.py>`_.
There, you can e.g. find the epoch- and batch-wise training loop. In the code block below, we show the constructor of TorchModel.

    .. code-block::

        class TorchModel(_base_model.BaseModel, abc.ABC):
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

We define attributes and suggest hyperparameters that are relevant for all neural network implementations,
e.g. the ``optimizer`` to use and the ``learning_rate`` to apply.
Some attributes are also set to fixed values, for instance the loss function (``self.loss_fn``) depending on the detected machine learning task.
Furthermore, early stopping is parametrized, which we use as a measure to prevent overfitting. With early stopping,
the validation loss is monitored and if it does not improve for a certain number of epochs (``self.early_stopping_patience``),
the training process is stopped. When working with our CNN implementation, it is important to keep in mind
that some relevant code and hyperparameters can also be found in TorchModel.

The definition of the CNN model itself as well as of some specific hyperparameters and ranges can be found in the `Cnn class <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/cnn.py>`_.
In the code block below, we show its ``define_model()`` method. Our CNN model consists of ``n_layers`` of blocks, which
include a ``Conv1d()``, ``BatchNorm()`` and ``Dropout`` layer.
The last of these blocks is followed by a ``MaxPool1d`` and ``Flatten`` layer.
This output is further processed by a ``Linear``, ``BatchNorm`` and ``Dropout`` layer, before the final  ``Linear()`` output layer.
The ``kernel_size`` and ``stride`` of the convolutional layers are two important hyperparameters that are optimized.
The number of output channels of the first layer (``out_channels``) and the frequency of doubling the number of output channels (``frequency_out_channels_doubling``) are currently
set, but can also be defined as an hyperparameter.
The number of outputs in the first linear layer after the convolutional blocks is defined by a hyperparameter (``n_initial_units_factor``),
that is multiplied with current dimensionality.
Further, we use ``Dropout`` for regularization and define the dropout rate as the hyperparameter ``p``.
Finally, we transform the list to which we added all network layers into a ``torch.nn.Sequential()`` object.

    .. code-block::

        def define_model(self) -> torch.nn.Sequential:
            """
            Definition of a CNN network.

            Architecture:

                - N_LAYERS of (Conv1d + BatchNorm + Dropout)
                - MaxPool1d, Flatten, Linear, BatchNorm, Dropout
                - Linear output layer

            Kernel sizes for convolutional and max pooling layers may be fixed or optimized.
            Same applies for strides, number of output channels of the first convolutional layer, dropout rate,
            frequency of a doubling of the output channels and number of units in the first linear layer.
            """
            n_layers = self.suggest_hyperparam_to_optuna('n_layers')
            model = []
            act_function = self.get_torch_object_for_string(string_to_get=self.suggest_hyperparam_to_optuna('act_function'))
            in_channels = self.width_onehot
            kernel_size = self.suggest_hyperparam_to_optuna('kernel_size')
            stride = max(1, int(kernel_size * self.suggest_hyperparam_to_optuna('stride_perc_of_kernel_size')))
            out_channels = 2 ** 2  # self.suggest_hyperparam_to_optuna('initial_out_channels_exp')
            frequency_out_channels_doubling = 2  # self.suggest_hyperparam_to_optuna('frequency_out_channels_doubling')
            p = self.suggest_hyperparam_to_optuna('dropout')
            for layer in range(n_layers):
                model.append(torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                             kernel_size=kernel_size, stride=stride))
                model.append(act_function)
                model.append(torch.nn.BatchNorm1d(num_features=out_channels))
                model.append(torch.nn.Dropout(p))
                in_channels = out_channels
                if ((layer+1) % frequency_out_channels_doubling) == 0:
                    out_channels *= 2
            model.append(torch.nn.MaxPool1d(kernel_size=kernel_size))
            model.append(torch.nn.Flatten())
            in_features = torch.nn.Sequential(*model)(torch.zeros(size=(1, self.width_onehot, self.n_features))).shape[1]
            out_features = int(in_features * self.suggest_hyperparam_to_optuna('n_units_factor_linear_layer'))
            model.append(torch.nn.Linear(in_features=in_features, out_features=out_features))
            model.append(act_function)
            model.append(torch.nn.BatchNorm1d(num_features=out_features))
            model.append(torch.nn.Dropout(p))
            model.append(torch.nn.Linear(in_features=out_features, out_features=self.n_outputs))
            return torch.nn.Sequential(*model)

The implementations for ``'classification'`` and ``'regression'`` just differ by the ``out_features`` of the output layer (and loss function as you can see in the first code block).
``self.n_outputs`` is inherited from ``BaseModel``, where it is set to 1 for ``regression`` (one continuous output)
or to the number of different classes for ``classification``.

**References**

1. Bishop, Christopher M. (2006). Pattern recognition and machine learning. New York, Springer.
2. Goodfellow, I., Bengio, Y.,, Courville, A. (2016). Deep Learning. MIT Press. Available at https://www.deeplearningbook.org/
