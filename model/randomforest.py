import sklearn

from model import sklearn_model


class RandomForest(sklearn_model.SklearnModel):
    """See BaseModel for more information on the parameters"""
    standard_encoding = '012'
    possible_encodings = ['012', 'raw']

    def define_model(self):
        """See BaseModel for more information"""
        # all hyperparameters defined are suggested for optimization
        params = self.suggest_all_hyperparams_to_optuna()
        params.update({'random_state': 42})
        if self.task == 'classification':
            return sklearn.ensemble.RandomForestClassifier(**params)
        else:
            return sklearn.ensemble.RandomForestRegressor(**params)

    def define_hyperparams_to_tune(self) -> dict:
        """See BaseModel for more information on the format"""
        return {
            'n_estimators': {
                'datatype': 'int',
                'lower_bound': 10,
                'upper_bound': 10000,
                'log': True
            },
            'max_depth': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 10
            },
            'min_samples_split': {
                'datatype': 'float',
                'lower_bound': 0.01,
                'upper_bound': 1
            },
            'min_samples_leaf': {
                'datatype': 'float',
                'lower_bound': 0.01,
                'upper_bound': 0.5
            },
            'max_leaf_nodes': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 100
            }
        }
