import optuna
import xgboost
import numpy as np

from model import base_model


class XgBoost(base_model.BaseModel):
    """See BaseModel for more information on the parameters"""
    standard_encoding = '012'
    possible_encodings = ['012', 'nuc']
    name = 'XGBoost'

    def __init__(self, task: str, optuna_trial: optuna.trial.Trial, encoding: str = None):
        super().__init__(task=task, optuna_trial=optuna_trial, encoding=encoding)

    def define_model(self) -> xgboost.XGBModel:
        """See BaseModel for more information"""
        # all hyperparameters defined for XGBoost are suggested for optimization
        if self.task == 'classification':
            return xgboost.XGBClassifier(**self.suggest_all_hyperparams_to_optuna())
        else:
            return xgboost.XGBRegressor(**self.suggest_all_hyperparams_to_optuna())

    def define_hyperparams_to_tune(self) -> dict:
        """See BaseModel for more information on the format"""
        return {
            'n_estimators': {
                'datatype': 'int',
                'lower_bound': 10,
                'upper_bound': 10000,
                'log': True
            },
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

    def train(self, X_train: np.array, y_train: np.array): #TODO: check if these can be unified in BaseModel as sklearn api is always the same
        """See BaseModel for more information"""
        self.model.fit(X_train, y_train)

    def predict(self, X_in: np.array) -> np.array:
        """See BaseModel for more information"""
        return self.model.predict(X_in)


