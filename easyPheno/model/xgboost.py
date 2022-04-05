import xgboost

from . import _sklearn_model


class XgBoost(_sklearn_model.SklearnModel):
    """
    Implementation of a class for XGBoost.

    See :obj:`~easyPheno.model._base_model.BaseModel` for more information on the attributes.
    """
    standard_encoding = '012'
    possible_encodings = ['012']

    def define_model(self) -> xgboost.XGBModel:
        """
        Definition of the actual prediction model.

        See :obj:`~easyPheno.model._base_model.BaseModel` for more information.
        """
        # all hyperparameters defined for XGBoost are suggested for optimization
        params = self.suggest_all_hyperparams_to_optuna()
        # add random_state for reproducibility
        params.update({'random_state': 42})
        if self.task == 'classification':
            # set some parameters to prevent warnings
            params.update({'use_label_encoder': False})
            eval_metric = 'mlogloss' if self.n_outputs > 2 else 'logloss'
            params.update({'eval_metric': eval_metric})
            return xgboost.XGBClassifier(**params)
        else:
            return xgboost.XGBRegressor(**params)

    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~easyPheno.model._base_model.BaseModel` for more information on the format.
        """
        return {
            'n_estimators': {
                'datatype': 'categorical',
                'list_of_values': [50, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 5000]
            },
            'learning_rate': {
                    'datatype': 'float',
                    'lower_bound': 0.05,
                    'upper_bound': 0.5,
                    'step': 0.05
            },
            'max_depth': {
                'datatype': 'int',
                'lower_bound': 3,
                'upper_bound': 10
            },
            'gamma': {
                'datatype': 'int',
                'lower_bound': 0,
                'upper_bound': 1000,
                'step': 10
            },
            'subsample': {
                'datatype': 'float',
                'lower_bound': 0.05,
                'upper_bound': 0.7,
                'step': 0.05
            },
            'colsample_bytree': {
                'datatype': 'float',
                'lower_bound': 0.05,
                'upper_bound': 0.7,
                'step': 0.05
            },
            'reg_lambda': {
                'datatype': 'float',
                'lower_bound': 0,
                'upper_bound': 1000,
                'step': 10
            },
            'reg_alpha': {
                'datatype': 'float',
                'lower_bound': 0,
                'upper_bound': 1000,
                'step': 10
            }
        }
