Multilayer Perceptron
===============================



    .. code-block::

        def define_model(self) -> torch.nn.Sequential:
            """
            Definition of an MLP network.

            Architecture:

                - N_LAYERS of (Linear + BatchNorm + Dropout)
                - Linear output layer

            Number of units in the first linear layer and percentage decrease after each may be fixed or optimized.
            """
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