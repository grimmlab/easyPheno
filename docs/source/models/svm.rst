Support Vector Machine / Regression
=============================================
Subsequently, we give details on the Support Vector Machine (SVM) respective Support Vector Regression (SVR) integrated in easyPheno.
Depending on the machine learning task that was detected (``'classification'`` or ``'regression'``), easyPheno automatically
switches between SVM and SVR.
References for a more detailed theoretical background can be found at the end of this page, which were also used for writing this page.
For our implementation, we use the machine learning framework scikit-learn, which also provides a `user guide for these models <https://scikit-learn.org/stable/modules/svm.html>`_.

**Support Vector Machine**

A SVM aims to find a hyperplane that optimally separates samples belonging to different classes, i.e. a hyperplane reflecting the maximum margin between the classes.
To address the non-linear-separable case, SVM introduces two of its main concepts, namely soft margin and the kernel trick.

With a so-called soft margin, mis-classifications are tolerated up to a certain degree.
Mathematically, this is realized by introducing a penalty term consisting of the so-called slack variables, which give the distance to the corresponding class margin and are set to zero in case of a correct classification.
The influence of this penalty term - and consequently "the degree of tolerance" - is controlled by a weighting factor usually called :math:`C`,
which is an important hyperparameter.
Larger values for :math:`C` lead to a stronger penalization of model errors, which might lead to a more accurate model, but is also more prone to overfitting.
In contrast, smaller :math:`C` values put less emphasis on wrong predictions. This has a regularizing effect lowering the risk of overfitting, but might also lead to underfitting.

The so-called kernel trick enables a transformation via kernel functions into a higher-dimensional space and consequently to find a solution for the separating hyperplane there.
Hence, the selection of the kernel function to use as well as the related hyperparameters are important during model training.

In the code bock below, you can see our implementation for the SVM respective SVR, which is implemented in the same class
and chosen based on the determined task (``if self.task == 'classification': ... else: ...``).
Furthermore, one can see that depending on the suggested ``kernel``, we are deciding which further hyperparameters need to be suggested.

    .. code-block::

        class SupportVectorMachine(_sklearn_model.SklearnModel):
            """
            Implementation of a class for Support Vector Machine respective Regression.

            See :obj:`~easypheno.model._base_model.BaseModel` for more information on the attributes.
            """
            standard_encoding = '012'
            possible_encodings = ['012']

            def define_model(self):
                """
                Definition of the actual prediction model.

                See :obj:`~easypheno.model._base_model.BaseModel` for more information.
                """
                kernel = self.suggest_hyperparam_to_optuna('kernel')
                reg_c = self.suggest_hyperparam_to_optuna('C')
                if kernel == 'poly':
                    degree = self.suggest_hyperparam_to_optuna('degree')
                    gamma = self.suggest_hyperparam_to_optuna('gamma')
                elif kernel in ['rbf', 'sigmoid']:
                    degree = 42  # default
                    gamma = self.suggest_hyperparam_to_optuna('gamma')
                elif kernel == 'linear':
                    degree = 42  # default
                    gamma = 42  # default
                if self.task == 'classification':
                    return sklearn.svm.SVC(kernel=kernel, C=reg_c, degree=degree, gamma=gamma, random_state=42,
                                           max_iter=1000000)
                else:
                    return sklearn.svm.SVR(kernel=kernel, C=reg_c, degree=degree, gamma=gamma, max_iter=1000000)

            def define_hyperparams_to_tune(self) -> dict:
                """
                See :obj:`~easypheno.model._base_model.BaseModel` for more information on the format.
                """
                return {
                    'kernel': {
                        'datatype': 'categorical',
                        'list_of_values': ['linear', 'poly', 'rbf'],
                    },
                    'degree': {
                        'datatype': 'int',
                        'lower_bound': 1,
                        'upper_bound': 5
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





**Support Vector Regression**

The concept of SVR is pretty similar, but instead of optimizing for a separating hyperplane,
the goal is to find a function that is within a certain threshold around the target values of the training samples.
Apart from that, similar concepts such as the kernel-trick are used, leading to similar hyperparameters that need to be optimized.

As already mentioned, you can find our implementation in the code block above, as SVM and SVR are implemented in the same class.

**References**

1. Bishop, Christopher M. (2006). Pattern recognition and machine learning. New York, Springer.
2. Smola, A. J. and Schölkopf, B. (2004). A tutorial on support vector regression. Statistics and computing 14, 199–222.
3. Drucker, H., Chris, Kaufman, B. L., Smola, A., and Vapnik, V. (1997). Support vector regression machines. In Advances in Neural Information Processing Systems 9. vol. 9, 155–161.
