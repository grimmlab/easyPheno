import sklearn

from model import sklearn_model


class SupportVectorMachine(sklearn_model.SklearnModel):
    """See BaseModel for more information on the parameters"""
    standard_encoding = '012'
    possible_encodings = ['012', 'raw']

    def define_model(self):
        """See BaseModel for more information"""
        # all hyperparameters defined are suggested for optimization
        kernel = self.suggest_hyperparam_to_optuna('kernel')
        reg_C = self.suggest_hyperparam_to_optuna('C')
        if kernel == 'poly':
            degree = self.suggest_hyperparam_to_optuna('degree')
        else:
            degree = 42  # default
        if kernel in ['rbf', 'poly', 'sigmoid']:
            gamma = self.suggest_hyperparam_to_optuna('gamma')
        else:
            gamma = 42  # default
        if self.task == 'classification':
            return sklearn.svm.SVC(kernel=kernel, C=reg_C, degree=degree, gamma=gamma, random_state=42)
        else:
            return sklearn.svm.SVR(kernel=kernel, C=reg_C, degree=degree, gamma=gamma)

    def define_hyperparams_to_tune(self) -> dict:
        """See BaseModel for more information on the format"""
        return {
            'kernel': {
                'datatype': 'categorical',
                'list_of_values': ['linear', 'poly', 'rbf', 'sigmoid']
            },
            'degree': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 3
            },
            'gamma': {
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
