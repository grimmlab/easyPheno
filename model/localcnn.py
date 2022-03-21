import tensorflow as tf

from model import _tensorflow_model


class LocalCnn(_tensorflow_model.TensorflowModel):
    standard_encoding = 'onehot'
    possible_encodings = ['onehot']

    def define_model(self) -> tf.keras.Sequential:
        """See BaseModel for more information"""
        n_layers = self.suggest_hyperparam_to_optuna('n_layers')
        model = tf.keras.Sequential()
        act_function = tf.keras.layers.Activation(self.suggest_hyperparam_to_optuna('act_function'))
        in_channels = self.width_onehot
        width = self.n_features
        model.add(tf.keras.Input(shape=(width, in_channels)))
        n_filters = 1
        kernel_size = 2 ** self.suggest_hyperparam_to_optuna('kernel_size_exp')
        stride = max(1, int(kernel_size * self.suggest_hyperparam_to_optuna('stride_perc_of_kernel_size')))
        model.add(tf.keras.layers.LocallyConnected1D(filters=n_filters, kernel_size=kernel_size,
                                                     strides=stride, activation=None))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(act_function)
        p = self.suggest_hyperparam_to_optuna('dropout')
        model.add(tf.keras.layers.Dropout(rate=p, seed=42))
        # Max pooling
        kernel_size_max_pool = 2 ** 4  # self.suggest_hyperparam_to_optuna('maxpool_kernel_size_exp')
        model.add(tf.keras.layers.MaxPool1D(pool_size=kernel_size_max_pool))
        # Flatten
        model.add(tf.keras.layers.Flatten())
        n_units = int(model.output_shape[1] * self.suggest_hyperparam_to_optuna('n_initial_units_factor'))
        for layer in range(n_layers):
            model.add(tf.keras.layers.Dense(units=n_units, activation=None))
            model.add(act_function)
            model.add(tf.keras.layers.BatchNormalization())
            p = self.suggest_hyperparam_to_optuna('dropout')
            model.add(tf.keras.layers.Dropout(rate=p))
            n_units = int(n_units * (1-self.suggest_hyperparam_to_optuna('perc_decrease_per_layer')))
        model.add(tf.keras.layers.Dense(units=self.n_outputs))
        return model

    def define_hyperparams_to_tune(self) -> dict:
        """See BaseModel for more information on the format"""
        return {
            'n_layers': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 3
            },
            'kernel_size_exp': {
                'datatype': 'int',
                'lower_bound': 4,
                'upper_bound': 8
            },
            'maxpool_kernel_size_exp': {
                'datatype': 'int',
                'lower_bound': 2,
                'upper_bound': 4
            },
            'stride_perc_of_kernel_size': {
                'datatype': 'categorical',
                'list_of_values': [0.5, 1]
            },
            'n_initial_units_factor': {
                'datatype': 'float',
                'lower_bound': 0.4,
                'upper_bound': 1,
                'step': 0.2
            },
            'perc_decrease_per_layer': {
                'datatype': 'float',
                'lower_bound': 0.1,
                'upper_bound': 0.5,
                'step': 0.2
            }
        }
