import optuna
import xgboost

from model import base_model


class XgBoost(base_model.BaseModel):
    """See BaseModel for more information on the parameters"""
    standard_encoding = '012'
    possible_encodings = ['012', 'nuc']
    name = 'XGBoost'

    def __init__(self, task: str, optuna_trial: optuna.trial.Trial, encoding: str = None):
        super().__init__(task=task, optuna_trial=optuna_trial, encoding=encoding)

    def define_model(self):
        if self.task == 'classification':
            return xgboost.XGBClassifier(**self.suggest_all_hyperparams_to_optuna())
        else:
            return xgboost.XGBRegressor(**self.suggest_all_hyperparams_to_optuna())

    def define_hyperparams_to_tune(self) -> dict:
        """See BaseModel for more information on the format"""
        return {
            'learning_rate': {
                    'datatype': 'float',
                    'lower_bound': 0,
                    'upper_bound': 0.5,
            },
            'gamma': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 10000,
                'log': True
            },
            'max_depth': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 10,
            },
            'subsample': {
                'datatype': 'float',
                'lower_bound': 0.01,
                'upper_bound': 1
            },
            'colsample_bytree': {
                'datatype': 'float',
                'lower_bound': 0.01,
                'upper_bound': 1
            },
            'colsample_bylevel': {
                'datatype': 'float',
                'lower_bound': 0.01,
                'upper_bound': 1
            },
            'colsample_bynode': {
                'datatype': 'float',
                'lower_bound': 0.01,
                'upper_bound': 1
            },
            'reg_lambda': {
                'datatype': 'float',
                'lower_bound': 0,
                'upper_bound': 1000,
            },
            'reg_alpha': {
                'datatype': 'float',
                'lower_bound': 0,
                'upper_bound': 1000,
            }
        }
