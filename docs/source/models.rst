Prediction Models
===================
easyPheno includes various phenotype prediction models, both classical genomic selection approaches as well as machine and deep learning-based methods.
In the following pages, we will give some details for all of the currently implemented models.

We provide both a workflow running easyPheno with a command line interface using Docker and as a pip package, see the following tutorials for more details:

    - :ref:`HowTo: Run easyPheno using Docker`

    - :ref:`HowTo: Use easyPheno as a pip package`

In both cases, you need to select the prediction model you want to run - or also multiple ones within the same optimization run.
A specific prediction model can be selected by giving the name of the *.py* file in which it is implemented (without the *.py* suffix).
For instance, if you want to run a Support Vector Machine implemented in *svm.py*, you need to specify *svm*.

easyPheno automatically chooses based on the selected phenotype whether to use the implementation for a classification (discrete trait) or regression (continuous trait) task.
All models except the classical genomic selection approaches (RR-BLUP and Bayesian alphabet models) provide an implementation for both cases.

In the following table, we give the keys for all prediction models as well as links to detailed descriptions and the source code:

.. list-table:: Phenotype Prediction Models
   :widths: 25 15 20 20 20
   :header-rows: 1

   * - Model
     - Key in easyPheno
     - Description
     - Source Code
     - Notes
   * - Ridge Regression BLUP
     - blup
     - :ref:`RR-BLUP`
     - `blup.py <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/blup.py>`_
     -
   * - Bayes A
     - bayesAfromR
     - :ref:`Bayesian alphabet`
     - `bayesAfromR.py <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/bayesAfromR.py>`_
     - requires Docker workflow
   * - Bayes B
     - bayesBfromR
     - :ref:`Bayesian alphabet`
     - `bayesBfromR.py <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/bayesBfromR.py>`_
     - requires Docker workflow
   * - Bayes C
     - bayesCfromR
     - :ref:`Bayesian alphabet`
     - `bayesCfromR.py <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/bayesCfromR.py>`_
     - requires Docker workflow
   * - L1-regularized Linear / Logistic Regression
     - linearregression
     - :ref:`Linear and Logistic Regression`
     - `linearregression.py <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/linearregression.py>`_
     - Regularization type in source code adjustable
   * - Elastic Net-regularized Linear / Logistic Regression
     - elasticnet
     - :ref:`Linear and Logistic Regression`
     - `elasticnet.py <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/elasticnet.py>`_
     -
   * - Support Vector Machine / Regression
     - svm
     - :ref:`Support Vector Machine / Regression`
     - `svm.py <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/svm.py>`_
     -
   * - Random Forest
     - randomforest
     - :ref:`Random Forest`
     - `randomforest.py <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/randomforest.py>`_
     -
   * - XGBoost
     - xgboost
     - :ref:`XGBoost`
     - `xgboost.py <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/xgboost.py>`_
     -
   * - Multilayer Perceptron
     - mlp
     - :ref:`Multilayer Perceptron`
     - `mlp.py <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/mlp.py>`_
     -
   * - Convolutional Neural Network
     - cnn
     - :ref:`Convolutional Neural Network`
     - `cnn.py <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/cnn.py>`_
     -
   * - Local Convolutional Neural Network
     - localcnn
     - :ref:`Local Convolutional Neural Network`
     - `localcnn.py <https://github.com/grimmlab/easyPheno/blob/main/easypheno/model/localcnn.py>`_
     -


If you are interested in adjusting an existing model or its hyperparameters: :ref:`HowTo: Adjust existing prediction models and their hyperparameters`.

If you want to integrate your own prediction model: :ref:`HowTo: Integrate your own prediction model`.

.. toctree::
    :maxdepth: 4
    :hidden:

    models/blup
    models/bayesianalphabet
    models/linreg
    models/svm
    models/rf
    models/xgb
    models/mlp
    models/cnn
    models/localcnn