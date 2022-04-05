import sklearn

from . import _sklearn_model


class RandomForest(_sklearn_model.SklearnModel):
    """
    Implementation of a class for Random Forest.

    See :obj:`~easyPheno.model._base_model.BaseModel` for more information on the attributes.
    """
    standard_encoding = '012'
    possible_encodings = ['012']

    def define_model(self):
        """
        Definition of the actual prediction model.

        See :obj:`~easyPheno.model._base_model.BaseModel` for more information.
        """
        # all hyperparameters defined are suggested for optimization
        params = self.suggest_all_hyperparams_to_optuna()
        # add random_state for reproducibility and n_jobs for multiprocessing
        params.update({'random_state': 42, 'n_jobs': -1})
        if self.task == 'classification':
            return sklearn.ensemble.RandomForestClassifier(**params)
        else:
            return sklearn.ensemble.RandomForestRegressor(**params)

    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~easyPheno.model._base_model.BaseModel` for more information on the format.
        """
        return {
            'n_estimators': {
                'datatype': 'categorical',
                'list_of_values': [50, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 5000]
            },
            'max_depth': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 10,
                'step': 1
            },
            'min_samples_split': {
                'datatype': 'float',
                'lower_bound': 0.1,
                'upper_bound': 1,
                'step': 0.1
            },
            'min_samples_leaf': {
                'datatype': 'float',
                'lower_bound': 0.1,
                'upper_bound': 0.5,
                'step': 0.1
            },
            'max_leaf_nodes': {
                'datatype': 'int',
                'lower_bound': 5,
                'upper_bound': 100,
                'step': 5
            }
        }
