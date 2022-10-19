Local Convolutional Neural Network
=============================================

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

