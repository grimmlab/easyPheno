HowTo: Adjust existing prediction models and their hyperparameters
==========================================================================
Every easyPheno prediction model based on `BaseModel <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/_base_model.py>`_
needs to implement several methods. Most of them are already implemented in `SklearnModel <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/_sklearn_model.py>`_,
`TorchModel <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/_torch_model.py>`_ and `TensorflowModel <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/_tensorflow_model.py>`_.
So if you make use of these, a prediction model only has to implement `define_model() <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/_base_model.py#L71>`_ and `define_hyperparams_to_tune() <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/_base_model.py#L88>`_.
We will therefore focus on these two methods in this tutorial.

If you want to create your own model, see :ref:`HowTo: Integrate your own prediction model`.

We already integrated several prediction models (see :ref:`Prediction Models`),
e.g. `LinearRegression <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/linearregression.py>`_
and `Mlp <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/mlp.py>`_, which we will use
for demonstration purposes in this HowTo.

Besides the written documentation, we recorded the tutorial video shown below with similar content.

Adjust prediction model
""""""""""""""""""""""""""
If you want to adjust the prediction model itself, you can change its definition in its implementation of ``define_model()``.
Let's discuss an example using `LinearRegression <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/linearregression.py>`_:

    .. code-block::

        def define_model(self):
            # Penalty term is fixed to l1, but might also be optimized
            penalty = 'l1'  # self.suggest_hyperparam_to_optuna('penalty')
            if penalty == 'l1':
                l1_ratio = 1
            elif penalty == 'l2':
                l1_ratio = 0
            else:
                l1_ratio = self.suggest_hyperparam_to_optuna('l1_ratio')
            if self.task == 'classification':
                reg_c = self.suggest_hyperparam_to_optuna('C')
                return sklearn.linear_model.LogisticRegression(penalty=penalty, C=reg_c, solver='saga',
                                                               l1_ratio=l1_ratio if penalty == 'elasticnet' else None,
                                                               max_iter=10000, random_state=42, n_jobs=-1)
            else:
                alpha = self.suggest_hyperparam_to_optuna('alpha')
                return sklearn.linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=42)

You can change the ``penalty`` term that is actually used by setting the related variable to a fixed value or suggest it as a hyperparameter for tuning (see below for information on how to add or adjust a hyperparameter or its range).
The same applies for further hyperparameters such as ``reg_c`` and ``alpha``. Beyond that, you could also adjust currently fixed parameters such as ``max_iter``.

Another example can be found in `Mlp <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/mlp.py>`_:

    .. code-block::

        def define_model(self) -> torch.nn.Sequential:
            n_layers = self.suggest_hyperparam_to_optuna('n_layers')
            model = []
            act_function = self.get_torch_object_for_string(string_to_get=self.suggest_hyperparam_to_optuna('act_function'))
            in_features = self.n_features
            out_features = int(in_features * self.suggest_hyperparam_to_optuna('n_initial_units_factor'))
            p = self.suggest_hyperparam_to_optuna('dropout')
            perc_decrease = self.suggest_hyperparam_to_optuna('perc_decrease_per_layer')
            for layer in range(n_layers):
                model.append(torch.nn.Linear(in_features=in_features, out_features=out_features))
                model.append(act_function)
                model.append(torch.nn.BatchNorm1d(num_features=out_features))
                model.append(torch.nn.Dropout(p=p))
                in_features = out_features
                out_features = int(in_features * (1-perc_decrease))
            model.append(torch.nn.Linear(in_features=in_features, out_features=self.n_outputs))
            return torch.nn.Sequential(*model)

Currently, the model consists of ``n_layers`` of a sequence of a ``Linear()``, ``BatchNorm()`` and ``Dropout()`` layer, finally followed by a ``Linear()`` output layer.
You can easily adjust this by e.g. adding further layers or setting ``n_layers`` to a fixed value.
Furthermore, the dropout rate ``p`` is optimized during hyperparameter search and the same rate is used for each Dropout layer.
You could set this to a fixed value or suggest a different value for each ``Dropout()`` layer
(e.g. by suggesting it via ``self.suggest_hyperparam_to_optuna('dropout')`` within the ``for``-loop).
Some hyperparameters are already defined in `TorchModel.common_hyperparams() <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/_torch_model.py#L196>`_ , which you can directly use here in its child class.
Furthermore, some of them are already suggested in `TorchModel <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/_torch_model.py>`_.

Beyond that, you can also change the complete architecture of the model if you prefer to do so,
e.g. by copying the file and adding your changes there (see also :ref:`HowTo: Integrate your own prediction model`).

Adjust hyperparameters
"""""""""""""""""""""""""
Besides changing the model definition, you can adjust the hyperparameters that are optimized as well as their ranges.
To set a hyperparameter to a fixed value, comment its suggestion and directly set a value, as described above.
If you want to optimize a hyperparameter which is currently set to a fixed value, do it the other way round.
If the hyperparameter is not yet defined in ``define_hyperparams_to_tune()``
(or ``common_hyperparams()`` in case of `TorchModel <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/_torch_model.py>`_
and `TensorflowModel <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/_tensorflow_model.py>`_),
you have to add it to ``define_hyperparams_to_tune()``.

Let's have a look at an example using `Mlp <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/mlp.py>`_:

    .. code-block::

        def define_hyperparams_to_tune(self) -> dict:
            n_layers = {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 5
            }
            n_initial_units_factor = {
                # Number of units in the first linear layer in relation to the number of inputs
                'datatype': 'float',
                'lower_bound': 0.1,
                'upper_bound': 0.7,
                'step': 0.05
            }
            perc_decrease_per_layer = {
                # Percentage decrease of the number of units per layer
                'datatype': 'float',
                'lower_bound': 0.1,
                'upper_bound': 0.5,
                'step': 0.05
            }
            if self.n_features > 20000:
                n_layers = {
                    'datatype': 'int',
                    'lower_bound': 1,
                    'upper_bound': 5
                }
                n_initial_units_factor = {
                    # Number of units in the first linear layer in relation to the number of inputs
                    'datatype': 'float',
                    'lower_bound': 0.1,
                    'upper_bound': 0.3,
                    'step': 0.01
                }
                perc_decrease_per_layer = {
                    # Percentage decrease of the number of units per layer
                    'datatype': 'float',
                    'lower_bound': 0.1,
                    'upper_bound': 0.5,
                    'step': 0.05
                }
            if self.n_features > 50000:
                n_layers = {
                    'datatype': 'int',
                    'lower_bound': 1,
                    'upper_bound': 3
                }
                n_initial_units_factor = {
                    # Number of units in the first linear layer in relation to the number of inputs
                    'datatype': 'float',
                    'lower_bound': 0.01,
                    'upper_bound': 0.15,
                    'step': 0.01
                }
                perc_decrease_per_layer = {
                    # Percentage decrease of the number of units per layer
                    'datatype': 'float',
                    'lower_bound': 0.2,
                    'upper_bound': 0.5,
                    'step': 0.05
                }

            return {
                'n_layers': n_layers,
                'n_initial_units_factor': n_initial_units_factor,
                'perc_decrease_per_layer': perc_decrease_per_layer
            }

There are multiple options to define a hyperparameter in easyPheno, see `define_hyperparams_to_tune() <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/_base_model.py#L88>`_ for more information regarding the format.
In the example above, three parameters are optimized depending on the number of features, besides the ones which are defined in the parent class TorchModel in `common_hyperparams() <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/_torch_model.py#L196>`_.
The method has to return a dictionary. So if you want to add a further hyperparameter, you need to add it to the dictionary with its name as the key and a dictionary defining its characteristics such as the ``datatype`` and ``lower_bound`` in case of a float or int as the value.
If you only want to change the range of an existing hyperparameter, you can just change the values in this method.



