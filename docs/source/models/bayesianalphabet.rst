Bayesian alphabet
=============================================
Subsequently, we give details on our implementation of models from the Bayesian alphabet, namely Bayes A, Bayes B and Bayes C.
These are classical genomic selection approaches that are closely related to each other.
References for a more detailed theoretical background can be found at the end of this page, which were also used for writing this text.

For our implementation, we used the R package `BGLR <https://cran.r-project.org/web/packages/BGLR/index.html>`_.
By doing so, we apply their efficient implementation and show the integration of an R package using `rpy2 <https://rpy2.github.io/>`_.
As a limitation, the Bayesian alphabet models are only available in the Docker workflow, as the R environment also needs to be
set up, which cannot be guaranteed for the Python package workflow. Furthermore, in the current implementation, they can only be used for continuous traits.

Similar to RR-BLUP, phenotype values :math:`\mathbf{y}` for Bayesian alphabet models are defined according to the following linear mixed model:

    .. math::
        \mathbf{y} = \mathbf{\beta} 1 + \mathbf{Xu} + \mathbf{\epsilon}

with the overall mean :math:`\mathbf{\beta}`, the genotype matrix :math:`\mathbf{X}` with corresponding
marker effects :math:`\mathbf{u}` and the residuals vector :math:`\mathbf{\epsilon}`.

In contrast to RR-BLUP, the variance of the residuals is commonly assigned a scaled-inverse Chi-squared distribution.
The difference between the models from the Bayesian alphabet is the prior distribution of the marker effects :math:`\mathbf{u}`:

    * Bayes A uses a scaled-*t* distribution.
    * Bayes B uses a mixture of two scaled-*t* distributions: one with a point of mass at zero and one with a large variance.
    * Bayes C uses a mixture of two normal distributions: one with a point of mass at zero and one with a large variance.

As mentioned above, we use the R package `BGLR <https://cran.r-project.org/web/packages/BGLR/index.html>`_ for our implementation.
To this end, we created a parent class `Bayes_R <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/_bayesfromR.py>`_
for all models from the Bayesian alphabet containing the integration of the R package.
In the code block below, we show the whole ``Bayes_R`` class.

    .. code-block::

        class Bayes_R(_param_free_base_model.ParamFreeBaseModel):
            standard_encoding = '012'
            possible_encodings = ['101']

            def __init__(self, task: str, model_name: str, encoding: str = None, n_iter: int = 6000, burn_in: int = 1000):
                super().__init__(task=task, encoding=encoding)
                self.model_name = model_name
                self.n_iter = n_iter
                self.burn_in = burn_in
                self.mu = None
                self.beta = None

            def fit(self, X: np.array, y: np.array) -> np.array:
                # import necessary R packages
                base = importr('base')
                BGLR = importr('BGLR')

                # create R objects for X and y
                R_X = robjects.r['matrix'](X, nrow=X.shape[0], ncol=X.shape[1])
                R_y = robjects.FloatVector(y)

                # run BGLR for BayesB
                ETA = base.list(base.list(X=R_X, model=self.model_name))
                fmBB = BGLR.BGLR(y=R_y, ETA=ETA, verbose=True, nIter=self.n_iter, burnIn=self.burn_in)

                # save results as numpy arrays
                self.beta = np.asarray(fmBB.rx2('ETA').rx2(1).rx2('b'))
                self.mu = fmBB.rx2('mu')
                return self.predict(X_in=X)

            def predict(self, X_in: np.array) -> np.array:
                return self.mu + np.matmul(X_in, self.beta)

The constructor has a parameter ``model_name``,
which we then use for switching between Bayes A, Bayes B and Bayes C. Furthermore, it contains ``n_iter`` and ``burn_in``,
so the number of total and burn in iterations. The whole model fitting can be found in the ``fit()`` method.
There, we first import R and the BGLR package. Then, we call the functions from BGLR to fit the model and
return the predicted values. For more information on the specific functions, we refer to the documentation of `BGLR <https://cran.r-project.org/web/packages/BGLR/index.html>`_.

The implementation of the Bayes A, Bayes B and Bayes C models is then straightforward. In the code block below, we examplary
show ``BayesA``. As you can see, the class is based on ``Bayes_R``. To select a specific method, we only need to
call the constructor of the parent class. All other methods are inherited from ``Bayes_R`` .

    .. code-block::

        class BayesA(_bayesfromR.Bayes_R):
            def __init__(self, task: str, encoding: str = None):
                super().__init__(task=task, model_name='BayesA', encoding=encoding)


**References**

1. Meuwissen, T. H., Hayes, B. J., & Goddard, M. E. (2001). Prediction of total genetic value using genome-wide dense marker maps. Genetics, 157(4), 1819–1829.
2. Habier, D., Fernando, R.L., Kizilkaya, K. et al. Extension of the bayesian alphabet for genomic selection. BMC Bioinformatics 12, 186 (2011)
3. Gianola D. (2013). Priors in whole-genome regression: the bayesian alphabet returns. Genetics, 194(3), 573–596.
