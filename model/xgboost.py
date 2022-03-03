import xgboost

from model import sklearn_model


class XgBoost(sklearn_model.SklearnModel):
    """See BaseModel for more information on the parameters"""
    standard_encoding = '012'
    possible_encodings = ['012', 'raw']

    def define_model(self) -> xgboost.XGBModel:
        """See BaseModel for more information"""
        # all hyperparameters defined for XGBoost are suggested for optimization
        if self.task == 'classification':
            params = self.suggest_all_hyperparams_to_optuna()
            params.update({'use_label_encoder': False})
            eval_metric = 'mlogloss' if self.n_outputs > 2 else 'logloss'
            params.update({'eval_metric': eval_metric})
            return xgboost.XGBClassifier(**params)
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
                    'step': 0.05
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
                'lower_bound': 0.05,
                'upper_bound': 1,
                'step': 0.05
            },
            'colsample_bytree': {
                'datatype': 'float',
                'lower_bound': 0.05,
                'upper_bound': 1,
                'step': 0.05
            },
            'colsample_bylevel': {
                'datatype': 'float',
                'lower_bound': 0.05,
                'upper_bound': 1,
                'step': 0.05
            },
            'colsample_bynode': {
                'datatype': 'float',
                'lower_bound': 0.05,
                'upper_bound': 1,
                'step': 0.05
            },
            'reg_lambda': {
                'datatype': 'float',
                'lower_bound': 0,
                'upper_bound': 1000
            },
            'reg_alpha': {
                'datatype': 'float',
                'lower_bound': 0,
                'upper_bound': 1000
            }
        }
