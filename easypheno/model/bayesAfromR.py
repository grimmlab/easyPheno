from . import _bayesfromR


class BayesA(_bayesfromR.Bayes_R):
    """
    Implementation of a class for Bayes A.

    *Attributes*

        *Inherited attributes*

        See :obj:`~easypheno.model._bayesfromR.Bayes_R` for more information on the attributes.
    """

    def __init__(self, task: str, encoding: str = None):
        super().__init__(task=task, model_name='BayesA', encoding=encoding)
