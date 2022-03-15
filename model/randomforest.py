import sklearn

from model import _sklearn_model


class RandomForest(_sklearn_model.SklearnModel):
    """See BaseModel for more information on the parameters"""
    standard_encoding = '012'
    possible_encodings = ['012']

    def define_model(self):
        """See BaseModel for more information"""
        # all hyperparameters defined are suggested for optimization
        params = self.suggest_all_hyperparams_to_optuna()
        params.update({'random_state': 42, 'n_jobs': -1})
        if self.task == 'classification':
            return sklearn.ensemble.RandomForestClassifier(**params)
        else:
            return sklearn.ensemble.RandomForestRegressor(**params)

    def define_hyperparams_to_tune(self) -> dict:
        """See BaseModel for more information on the format"""
        return {
            'n_estimators': {
                'datatype': 'categorical',
                'list_of_values': [10, 50, 100, 200, 300, 400, 500, 750, 1000]
            },
            'max_depth': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 10
            },
            'min_samples_split': {
                'datatype': 'float',
                'lower_bound': 0.01,
                'upper_bound': 1,
                'step': 0.05
            },
            'min_samples_leaf': {
                'datatype': 'float',
                'lower_bound': 0.01,
                'upper_bound': 0.5,
                'step': 0.05
            },
            'max_leaf_nodes': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 100,
                'step': 5
            }
        }
