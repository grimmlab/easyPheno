from . import _bayes_R


class BayesB(_bayes_R.Bayes_R):
    """
    Implementation of a class for Bayes B.

    *Attributes*

        *Inherited attributes*

        See :obj:`~easyPheno.model._bayes_R.Bayes_R` for more information on the attributes.
    """

    def __init__(self, task: str, encoding: str = None):
        super().__init__(task=task, model_name='BayesB', encoding=encoding)
