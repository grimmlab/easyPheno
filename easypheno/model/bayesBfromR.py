from . import _bayesfromR


class BayesB(_bayesfromR.Bayes_R):
    """
    Implementation of a class for Bayes B.

    *Attributes*

        *Inherited attributes*

        See :obj:`~easypheno.model._bayesfromR.Bayes_R` for more information on the attributes.
    """

    def __init__(self, task: str, encoding: str = None):
        super().__init__(task=task, model_name='BayesB', encoding=encoding)
