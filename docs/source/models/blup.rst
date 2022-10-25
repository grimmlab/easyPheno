RR-BLUP
=============================================
Subsequently, we give details on our implementation of Ridge Regression Best Linear Unbiased Predictor (RR-BLUP),
which is a classical genomic selection approach.
References for a more detailed theoretical background can be found at the end of this page, which were also used for writing this text.

RR-BLUP is based on a linear mixed model, for which phenotype values :math:`\mathbf{y}` can be calculated as

    .. math::
        \mathbf{y} = \mathbf{\beta} 1 + \mathbf{Xu} + \mathbf{\epsilon}

with the overall mean :math:`\mathbf{\beta}`, the genotype matrix :math:`\mathbf{X}` with corresponding
marker effects :math:`\mathbf{u}` and the residuals vector :math:`\mathbf{\epsilon}`.
When fitting the model to the training data, :math:`\mathbf{\beta} and `:math:`\mathbf{u}` are determined.

In easyPheno, RR-BLUP is implemented as a child class of `ParamFreeBaseModel <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/_param_free_base_model.py>`_
and is named `Blup <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/blup.py>`_.
As you can see there, the ``fit()`` method contains the fitting of the model to match the training data.
In its current implementation, RR-BLUP can only be used for continuous traits.

**References**

1. Meuwissen, T. H., Hayes, B. J., & Goddard, M. E. (2001). Prediction of total genetic value using genome-wide dense marker maps. Genetics, 157(4), 1819–1829.




