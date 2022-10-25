Random Forest
=============================================
Subsequently, we give details on our implementation of Random Forest.
Depending on the machine learning task that was detected (``'classification'`` or ``'regression'``), easyPheno automatically
switches between both implementations.
References for a more detailed theoretical background can be found at the end of this page, which were also used for writing this text.
For our implementation, we use the machine learning framework scikit-learn,
which also provides a `user guide for Random Forests <https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees>`_.

A Random Forest is a so-called ensemble learner, which uses multiple weak learners - in this case Decision Trees - to
derive a final prediction. Random Forests use a technique called Bootstrap aggregating, usually abbreviated with Bagging.
With this technique, a random subsample with replacement (bootstrap samples) of the whole training data is used to
construct each Decision Tree, and finally the predictions of these are aggregated. Beyond that,
the algorithm was extended by not using all features for training each weak learner, but also a random subset.
The goal of both techniques is to prevent overfitting, which Decision Trees tend to, by decreasing variance of the ensemble.

It is worth to mention that the individual weak learners are trained independent from each other, so their construction
can also be parallelized. In the end, the individual predictions need to be combined, for which several approaches exist.
For regression tasks, our implementation averages the predictions of all Decision Trees in the ensemble.
In case of a classification, the predictions of each weak learner are weighted by the probability estimates
and then averaged across the whole ensemble. Finally, the class with the largest averaged probability is predicted.

As you can see in the code block below, we use ``RandomForestClassifier`` respective ``RandomForestRegressor``
from scikit-learn. Besides the strategy for determining the ``max_features`` per Decision Tree,
we optimize hyperparameters such as the number of trees in the whole ensemble (``n_estimators``).

    .. code-block::

        class RandomForest(_sklearn_model.SklearnModel):
            """
            Implementation of a class for Random Forest.

            See :obj:`~easypheno.model._base_model.BaseModel` for more information on the attributes.
            """
            standard_encoding = '012'
            possible_encodings = ['012']

            def define_model(self):
                """
                Definition of the actual prediction model.

                See :obj:`~easypheno.model._base_model.BaseModel` for more information.
                """
                # all hyperparameters defined are suggested for optimization
                params = self.suggest_all_hyperparams_to_optuna()
                # add random_state for reproducibility and n_jobs for multiprocessing
                params.update({'random_state': 42, 'n_jobs': -1})
                if self.task == 'classification':
                    return sklearn.ensemble.RandomForestClassifier(**params)
                else:
                    return sklearn.ensemble.RandomForestRegressor(**params)

            def define_hyperparams_to_tune(self) -> dict:
                """
                See :obj:`~easypheno.model._base_model.BaseModel` for more information on the format.
                """
                return {
                    'n_estimators': {
                        'datatype': 'categorical',
                        'list_of_values': [50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000,
                                           3500, 4000, 4500, 5000]
                    },
                    'min_samples_split': {
                        'datatype': 'float',
                        'lower_bound': 0.005,
                        'upper_bound': 0.2,
                        'step': 0.005
                    },
                    'max_depth': {
                        'datatype': 'int',
                        'lower_bound': 2,
                        'upper_bound': 50,
                        'step': 2
                    },
                    'min_samples_leaf': {
                        'datatype': 'float',
                        'lower_bound': 0.005,
                        'upper_bound': 0.2,
                        'step': 0.005
                    },
                    'max_features': {
                        'datatype': 'categorical',
                        'list_of_values': ['sqrt', 'log2']
                    }
                }

For further information on all hyperparameters, we refer to the documentation of scikit-learn:
`Random Forest Classifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier>`_ and
`Random Forest Regressor <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor>`_.

**References**

1. Breiman, L. (2001). Random forests. Machine Learning 45, 5–32.
2. Hastie, T., Tibshirani, R., & Friedman, J. H. (2009). The elements of statistical learning: data mining, inference, and prediction. 2nd ed. New York, Springer.

