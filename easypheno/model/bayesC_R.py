from . import _bayes_R


class BayesC(_bayes_R.Bayes_R):
    """
    Implementation of a class for Bayes A.

    *Attributes*

        *Inherited attributes*

        See :obj:`~easypheno.model._bayes_R.Bayes_R` for more information on the attributes.
    """

    def __init__(self, task: str, encoding: str = None):
        super().__init__(task=task, model_name='BayesC', encoding=encoding)
