Local Convolutional Neural Network
=============================================
Subsequently, we give details on our implementation of a Local Convolutional Neural Network (LCNN).
References for a more detailed theoretical background can be found at the end of this page, which were also used for writing this page.
We use TensorFlow for our implementation. For more information on specific TensorFlow objects that we use,
e.g. layers, see the `TensorFlow documentation <https://www.tensorflow.org/api_docs/python/tf>`_.

In contrast to normal convolutional layers, local convolutional layers have region-specific filters with individual weights.
In the context of phenotype prediction, marker variants in different regions in the genome might have a completely different
influence on the phenotype. The hope is to capture these different effects via the region-specific filters of the local convolutional layers.

For LCNN, we one-hot encoded the data, as this data can be easily processed by a LCNN.
This type of encoding preserves the whole nucleotide information and might thus lead to a smaller information loss than other encodings.

Some of the methods and attributes relevant for the LCNN are already defined in its parent class `TensorflowModel <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/_tensorflow_model.py>`_.
There, you can e.g. find the epoch- and batch-wise training loop. In the code block below, we show the constructor of TensorflowModel.

    .. code-block::

        class TensorflowModel(_base_model.BaseModel, abc.ABC):
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
                self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.suggest_hyperparam_to_optuna('learning_rate'))
                self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) if task == 'classification' \
                    else tf.keras.losses.MeanSquaredError()
                # early stopping if there is no improvement on validation loss for a certain number of epochs
                self.early_stopping_patience = self.suggest_hyperparam_to_optuna('early_stopping_patience')
                self.early_stopping_point = early_stopping_point
                self.early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=self.early_stopping_patience, mode='min', restore_best_weights=True,
                    min_delta=0.1
                )
                self.model.compile(self.optimizer, loss=self.loss_fn)

We define attributes and suggest hyperparameters that are relevant for all neural network implementations,
e.g. the optimizer to use and learning rate to apply.
Some attributes are also set to fixed values, for instance the loss function depending on the detected machine learning task.
Furthermore, early stopping is parametrized, which we use as a measure to prevent overfitting. With early stopping,
the validation loss is monitored and if it does not improve for a certain number of epochs (``self.early_stopping_patience``),
the training process is stopped. When working with our LCNN implementation, it is important to keep in mind
that some relevant code and hyperparameters can also be found in TensorflowModel.

The definition of the model itself as well as of some specific hyperparameters and ranges can be found in the `LocalCnn class <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/cnn.py>`_.
In the code block below, we show its ``define_model()``. The architecture of our LCNN model starts with a
``LocallyConnected1D`` layer, for which the ``kernel_size`` and ``stride`` are optimized during hyperparameter search.
This layer is followed by a ``BatchNormalization``, ``Dropout``, ``MaxPool`` and ``Flatten`` layer.
This output is forwarded to ``n_layers`` of blocks, which include a ``Dense()``, ``BatchNormalization()`` and ``Dropout`` layer.
The last of these blocks is followed by a ``Dense`` output layer.
The number of outputs in the first ``Dense`` layer is defined by a hyperparameter (``n_initial_units_factor``),
that is multiplied with the number of inputs. Then, with each of the above-mentioned blocks, the number of outputs
decreases by a percentage parameter ``perc_decrease``.
Further, we use ``Dropout`` for regularization and define the dropout rate as the hyperparameter ``p``.

    .. code-block::

        def define_model(self) -> tf.keras.Sequential:
            """
            Definition of a LocalCNN network.

            Architecture:

                - LocallyConnected1D, BatchNorm, Dropout, MaxPool1D, Flatten
                - N_LAYERS of (Dense + BatchNorm + Dropout)
                - Dense output layer

            Kernel size for LocallyConnectedLayer and max pooling layer may be fixed or optimized.
            Same applies for stride, number of units in the first dense layer and percentage decrease after each layer.
            """
            n_layers = self.suggest_hyperparam_to_optuna('n_layers')
            model = tf.keras.Sequential()
            act_function = tf.keras.layers.Activation(self.suggest_hyperparam_to_optuna('act_function'))
            l1_regularizer = None  # tf.keras.regularizers.L1(l1=self.suggest_hyperparam_to_optuna('l1_factor'))
            in_channels = self.width_onehot
            width = self.n_features
            model.add(tf.keras.Input(shape=(width, in_channels)))
            n_filters = 1
            kernel_size = int(2 ** self.suggest_hyperparam_to_optuna('kernel_size_exp'))
            stride = max(1, int(kernel_size * self.suggest_hyperparam_to_optuna('stride_perc_of_kernel_size')))
            model.add(tf.keras.layers.LocallyConnected1D(filters=n_filters, kernel_size=kernel_size,
                                                         strides=stride, activation=None,
                                                         kernel_regularizer=l1_regularizer))
            model.add(act_function)
            model.add(tf.keras.layers.BatchNormalization())
            p = self.suggest_hyperparam_to_optuna('dropout')
            model.add(tf.keras.layers.Dropout(rate=p, seed=42))
            kernel_size_max_pool = 2 ** 4  # self.suggest_hyperparam_to_optuna('maxpool_kernel_size_exp')
            model.add(tf.keras.layers.MaxPool1D(pool_size=kernel_size_max_pool))
            model.add(tf.keras.layers.Flatten())
            n_units = int(model.output_shape[1] * self.suggest_hyperparam_to_optuna('n_initial_units_factor'))
            perc_decrease = self.suggest_hyperparam_to_optuna('perc_decrease_per_layer')
            for layer in range(n_layers):
                model.add(tf.keras.layers.Dense(units=n_units, activation=None,
                                                kernel_regularizer=l1_regularizer))
                model.add(act_function)
                model.add(tf.keras.layers.BatchNormalization())
                model.add(tf.keras.layers.Dropout(rate=p))
                n_units = int(n_units * (1-perc_decrease))
            model.add(tf.keras.layers.Dense(units=self.n_outputs))
            return model

The implementations for ``'classification'`` and ``'regression'`` just differ by the ``units`` of the output layer (and loss function as you can see in the first code block).
``self.n_outputs`` is inherited from ``BaseModel``, where it is set to 1 for ``regression`` (one continuous output)
or to the number of different classes for ``classification``.

**References**

1. Bishop, Christopher M. (2006). Pattern recognition and machine learning. New York, Springer.
2. Goodfellow, I., Bengio, Y.,, Courville, A. (2016). Deep Learning. MIT Press. Available at https://www.deeplearningbook.org/
3. Pook, T., Freudenthal, J.A., Korte, A., & Simianer, H. (2020). Using Local Convolutional Neural Networks for Genomic Prediction. Frontiers in Genetics, 11.