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
        # Add n_layers with: Conv1d + BatchNorm + activation + Dropout
        for layer in range(n_layers):
            out_channels = 2 ** self.suggest_hyperparam_to_optuna('out_channels_exp')
            kernel_size = 2 ** self.suggest_hyperparam_to_optuna('kernel_size_exp')
            stride = max(1, int(kernel_size * self.suggest_hyperparam_to_optuna('stride_perc_of_kernel_size')))
            model.add(tf.keras.layers.Conv1D(filters=out_channels, kernel_size=kernel_size,
                                             strides=stride, activation=None))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(act_function)
            p = self.suggest_hyperparam_to_optuna('dropout')
            model.add(tf.keras.layers.Dropout(rate=p, seed=42))
        # Max pooling
        kernel_size_max_pool = 2 ** self.suggest_hyperparam_to_optuna('kernel_size_exp')
        model.add(tf.keras.layers.MaxPool1D(pool_size=kernel_size_max_pool))
        # Flatten and linear layers with dropout
        model.add(tf.keras.layers.Flatten())
        out_features = 2 ** self.suggest_hyperparam_to_optuna('n_units_per_layer_exp')
        model.add(tf.keras.layers.Dense(units=out_features, activation=None))
        model.add(act_function)
        p = self.suggest_hyperparam_to_optuna('dropout')
        model.add(tf.keras.layers.Dropout(rate=p))
        model.add(tf.keras.layers.Dense(units=self.n_outputs))
        return model

    def define_hyperparams_to_tune(self) -> dict:
        """See BaseModel for more information on the format"""
        return { # TODO: ranges anpassen for start der Experimente
            'n_layers': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 4  # 10
            },
            'out_channels_exp': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 3
            },
            'kernel_size_exp': {
                'datatype': 'int',
                'lower_bound': 2,
                'upper_bound': 3  # 8
            },
            'stride_perc_of_kernel_size': {
                'datatype': 'float',
                'lower_bound': 0,
                'upper_bound': 1,
                'step': 0.1
            },
            'n_units_per_layer_exp': {
                'datatype': 'int',
                'lower_bound': 2,
                'upper_bound': 3  # 10
            }
        }
