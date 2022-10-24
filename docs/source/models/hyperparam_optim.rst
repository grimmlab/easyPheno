Hyperparameter optimization
===========================================
Subsequently, we give details on our implementation of Bayesian optimization for the automatic hyperparameter search.
References for a more detailed theoretical background can be found at the end of this page, which were also used for writing this page.
For our implementation, we use the optimization framework `Optuna <https://optuna.readthedocs.io/en/stable/>`_,
for which its developers also provide a comprehensive online documentation.

Common hyperparameter optimization methods, e.g. Grid Search or Random Search, do not make use of information gained
during the optimization process. However, Bayesian optimization uses this knowledge and tries to direct
the hyperparameter search towards more promising parameter candidates. The search is guided by a so-called objective value,
which is in the machine learning context usually the performance on validation data. With this objective value,
a probability model mapping from parameter candidates to a probability of an objective value can be defined. As a result,
the most promising parameters can be selected for further trials. This trial-wise optimization with using existing knowledge
makes Bayesian optimization potentially more efficient than Grid or Random Search, despite the computational resources
needed for the selection of parameter candidates.

Our implementation can be found in the class `OptunaOptim <https://github.com/grimmlab/easyPheno/blob/main/easypheno/optimization/optuna_optim.py>`_.
Besides results saving, the main part of this class can be found in the ``objective()`` method.
This method is called for each new trial. At the beginning of each new trial, a prediction model using the new parameter
set is defined. Then, we loop over the whole training and validation data to retrieve the objective value. In case of
multiple validation sets, we use the mean value.

To improve efficiency, we implemented pruning based on intermediate results - so results on validation sets within the cross-validation -
and stop a trial if the intermediate result is worse than the 80th percentile of previous ones at the same time.
The probability that such a parameter set would let to better results in the end is pretty low. Furthermore,
we set the number of finished trials before we start with pruning to 20.
Besides that, we check for parameter set duplicates, as the implementation of Optuna does not prevent to suggest
the same parameters again (if they are most likely the best ones to suggest in the current state).
The whole optimization process is saved in a database for debugging purposes.

For more detailed information regarding the objects and functions we use from Optuna, see the `Optuna documentation <https://optuna.readthedocs.io/en/stable/>`_.

**References**

1. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework.
2. Bergstra, J., Bardenet, Ré., Bengio, Y. & Kégl, B. (2011). Algorithms for Hyper-parameter Optimization.
3. Snoek, J., Larochelle, H. & Adams, R. P. (2012). Practical bayesian optimization of machine learning algorithms.





