Prediction Models
===================
easyPheno includes various phenotype prediction models, both classical genomic selection approaches as well as machine and deep learning-based methods.
In the following pages, we will give some details for all of the currently implemented models.

We provide both a workflow running easyPheno using Docker and as a pip package, see the following tutorials for more details:

    - :ref:`HowTo: Run easyPheno using Docker`

    - :ref:`HowTo: Use easyPheno as a pip package`

In both cases, you need to select the prediction model you want to run - or also multiple ones within the same optimization run.
A specific prediction model can be selected by giving the name of the *.py* file in which it is implemented (without the *.py* suffix)
For instance, if you want to run a Support Vector Machine implemented in *svm.py*, you need to specify *svm*.

In the following table, we give the keys for all prediction models as well as links to detailed descriptions and the source code:




If you are interested in adjusting an existing model or its hyperparameters: :ref:`HowTo: Adjust existing prediction models and their hyperparameters`.
If you want to integrate your own prediction model: :ref:`HowTo: Integrate your own prediction model`.



.. toctree::
    :maxdepth: 4

    models/blup
    models/bayesianalphabet
    models/linreg
    models/svm
    models/rf
    models/xgb
    models/mlp
    models/cnn
    models/localcnn