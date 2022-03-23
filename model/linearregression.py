import sklearn

from model import _sklearn_model


class LinearRegression(_sklearn_model.SklearnModel):
    """
    See BaseModel for more information on the attributes.
    """
    standard_encoding = '012'
    possible_encodings = ['012']

    def define_model(self):
        """
        See BaseModel for more information.
        """
        # Penalty term is fixed to l1, but might also be optimized
        penalty = 'l1'  # self.suggest_hyperparam_to_optuna('penalty')
        if penalty == 'l1':
            l1_ratio = 1
        elif penalty == 'l2':
            l1_ratio = 0
        else:
            l1_ratio = self.suggest_hyperparam_to_optuna('l1_ratio')
        if self.task == 'classification':
            reg_C = self.suggest_hyperparam_to_optuna('C')
            return sklearn.linear_model.LogisticRegression(penalty=penalty, C=reg_C, solver='saga',
                                                           l1_ratio=l1_ratio if penalty == 'elasticnet' else None,
                                                           max_iter=10000, random_state=42)
        else:
            alpha = self.suggest_hyperparam_to_optuna('alpha')
            return sklearn.linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=42)

    def define_hyperparams_to_tune(self) -> dict:
        """
        See BaseModel for more information on the format.
        """
        return {
            'penalty': {
                'datatype': 'categorical',
                'list_of_values': ['l1', 'l2', 'elasticnet']
            },
            'l1_ratio': {
                'datatype': 'float',
                'lower_bound': 0.05,
                'upper_bound': 0.95,
                'step': 0.05
            },
            'alpha': {
                'datatype': 'float',
                'lower_bound': 10**-3,
                'upper_bound': 10**3,
                'log': True
            },
            'C': {
                'datatype': 'float',
                'lower_bound': 10**-3,
                'upper_bound': 10**3,
                'log': True
            }
        }
