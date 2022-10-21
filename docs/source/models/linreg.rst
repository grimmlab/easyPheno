Linear and Logistic Regression
=============================================
Subsequently, we give details on the regularized linear respective logistic regression approaches that are integrated in easyPheno.
First, we outline the regularized linear regression models, which can be used for predicting continuous traits.
Then, we describe the closely related logistic regression approaches suitable for discrete phenotypes.
References for a more detailed theoretical background can be found at the end of this page, which were also used for writing this page.
For our implementation, we use the machine learning framework scikit-learn, which also provides a `user guide for these models <https://scikit-learn.org/stable/modules/linear_model.html>`_.

**Regularized linear regression for continuous traits**

We provide regularized linear regressions models, for which the model weights can be optimized by minimizing
the deviation between predicted and true phenotypic values, often with considering an additive penalty term for regularization:

    .. math::
       \mathrm{argmin}_{\mathbf{w}} \frac{1}{2} |\mathbf{y} - \mathbf{X^{\ast}} \mathbf{w} |_2^2 + \alpha \Omega(\mathbf{w})

In case of the Least Absolute Shrinkage and Selection Operator, usually abbreviated with LASSO,
the L1-norm, so the sum of the absolute value of the weights, is used for regularization. This constraint
usually leads to sparse solutions forcing unimportant weights to zero. Intuitively speaking, this can be seen as an automatic feature selection.
The L2-norm, also known as the Euclidean norm, is defined as the square root of the summed up quadratic weights.
Regularized linear regression using the L2-norm is usually called Ridge Regression. This penalty term has the effect
of grouping correlated features. Elastic Net combines both the L1- and the L2-norm, introducing a further hyperparameter
controlling the influence of each of the two parts.

All these three approaches - LASSO, Ridge and Elastic Net Regression - are currently implemented in easyPheno.
However, as the feature selection effect of LASSO seems to be profitable considering often large genotype matrices,
LASSO and Elastic Net Regression are activated by a variable for selecting the penalty term.

The following code block shows the implementation of LASSO in `linearregression.py <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/linearregression.py>`_.

    .. code-block::

        class LinearRegression(_sklearn_model.SklearnModel):
        """
        Implementation of a class for Linear respective Logistic Regression.

        See :obj:`~easypheno.model._base_model.BaseModel` for more information on the attributes.
        """
        standard_encoding = '012'
        possible_encodings = ['012']

        def define_model(self):
            """
            Definition of the actual prediction model.

            See :obj:`~easypheno.model._base_model.BaseModel` for more information.
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
                reg_c = self.suggest_hyperparam_to_optuna('C')
                return sklearn.linear_model.LogisticRegression(penalty=penalty, C=reg_c, solver='saga',
                                                               l1_ratio=l1_ratio if penalty == 'elasticnet' else None,
                                                               max_iter=10000, random_state=42, n_jobs=-1)
            else:
                alpha = self.suggest_hyperparam_to_optuna('alpha')
                return sklearn.linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=42)

        def define_hyperparams_to_tune(self) -> dict:
            """
            See :obj:`~easypheno.model._base_model.BaseModel` for more information on the format.
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

Currently, we set ``penalty='l1'`` in ``define_model()``, to get the implementation of LASSO (or L1-regularized Logistic Regression).
But one could also choose another penalty term or treat its selection as a hyperparameter, see :ref:`HowTo: Adjust existing prediction models and their hyperparameters`.

Furthermore, Elastic Net is implemented in a separate file containing very similar code to enable a comparison of Elastic Net and LASSO regression.
Its implementation can be found in `elasticnet.py <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/elasticnet.py>`_.


**Regularized logistic regression for discrete traits**

In contrast to linear regression that is applied for regression tasks (continuous traits), logistic regression is used for classification (discrete traits).
Logistic regression applies the logistic function to the linear combination of the features and weights to get probability scores and assign a discrete label.
The same penalty terms as for regularized linear regression (L1, L2 and Elastic Net) are often included in the cost function that is optimized during training,
with similar effects as described above.

We implemented regularized logistic regression in the same classes as linear regression (see `linearregression.py <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/linearregression.py>`_ and `elasticnet.py <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/elasticnet.py>`_) and switch between both based on the machine learning
task that was detected by easyPheno (see ``if self.task == 'classification': ... else: ...`` in the code block above).


**References**

1. Hastie, T., Tibshirani, R., & Friedman, J. H. (2009). The elements of statistical learning: data mining, inference, and prediction. 2nd ed. New York, Springer.
2. Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society: Series B (Methodological), 58(1), 267–288.
3. Zou, H. and Hastie, T. (2005). Regularization and variable selection via the elastic net. Journal of the Royal Statistical Society, Series B, 67, 301–320.
4. Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
