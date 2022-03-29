import sklearn

from model import _sklearn_model


class TestModel(_sklearn_model.SklearnModel):
    """
    See BaseModel for more information on the attributes.
    """
    standard_encoding = '012'
    possible_encodings = ['012']

    def define_model(self, parameter: str):
        """
        See BaseModel for more information.

        :param parameter: this is just a test

        """
        return 'Test'

    def define_hyperparams_to_tune(self) -> dict:
        """
        See BaseModel for more information on the format.
        """
        return {
            'kernel': {
                'datatype': 'categorical',
                'list_of_values': ['linear', 'poly', 'rbf']
            }
        }
