.. image:: https://raw.githubusercontent.com/grimmlab/easyPheno/main/docs/image/Logo_easyPheno_Text.png
    :width: 800
    :alt: easyPheno


easyPheno: A state-of-the-art and easy-to-use Python framework for plant phenotype prediction
=============================================================================================================
easyPheno is a Python framework that enables the rigorous training, comparison and analysis of phenotype predictions for a variety of different models.
easyPheno includes multiple state-of-the-art prediction models.
Besides common genomic selection approaches, such as best linear unbiased prediction (BLUP) and models from the Bayesian alphabet, our framework includes several machine learning methods.
These range from classical models, such as regularized linear regression over ensemble learners, e.g. XGBoost, to deep learning-based architectures, such as Convolutional Neural Networks (CNN).
To enable automatic hyperparameter optimization, we leverage  state-of-the-art and efficient Bayesian optimization techniques.
In addition, our framework is designed to allow an easy and straightforward integration of further prediction models.

easyPheno and its documentation is currently under construction.
The pip-package is not yet available, but feel free to already test the Docker workflow, see here: :ref:`HowTo: Run easyPheno using Docker`.


Contributors
----------------------------------------

This pipeline is developed and maintained by members of the `Bioinformatics lab <https://bit.cs.tum.de>`_ lead by `Prof. Dr. Dominik Grimm <https://bit.cs.tum.de/team/dominik-grimm/>`_:

- `Florian Haselbeck, M.Sc. <https://bit.cs.tum.de/team/florian-haselbeck/>`_
- `Maura John, M.Sc. <https://bit.cs.tum.de/team/maura-john/>`_

Citation
---------------------
When using this workflow, please cite our publication:

tbd.

.. toctree::
   :maxdepth: 4

   install
   tutorials
   data
