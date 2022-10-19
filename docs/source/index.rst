.. image:: https://raw.githubusercontent.com/grimmlab/easyPheno/main/docs/image/Logo_easyPheno_Text.png
    :width: 600
    :alt: easyPheno

|
easyPheno: A phenotype prediction framework
===================================================
easyPheno is a Python framework that enables the rigorous training, comparison and analysis of phenotype predictions for a variety of different models.
easyPheno includes multiple state-of-the-art prediction models.
Besides common genomic selection approaches, such as best linear unbiased prediction (BLUP) and models from the Bayesian alphabet, our framework includes several machine learning methods.
These range from classical models, such as regularized linear regression over ensemble learners, e.g. XGBoost, to deep learning-based architectures, such as Convolutional Neural Networks (CNN).
To enable automatic hyperparameter optimization, we leverage state-of-the-art and efficient Bayesian optimization techniques.
In addition, our framework is designed to allow an easy and straightforward integration of further prediction models.


Contributors
----------------------------------------

This pipeline is developed and maintained by members of the `Bioinformatics lab <https://bit.cs.tum.de>`_ lead by `Prof. Dr. Dominik Grimm <https://bit.cs.tum.de/team/dominik-grimm/>`_:

- `Florian Haselbeck, M.Sc. <https://bit.cs.tum.de/team/florian-haselbeck/>`_
- `Maura John, M.Sc. <https://bit.cs.tum.de/team/maura-john/>`_

Citation
---------------------
A manuscript for publishing easyPheno as a scientific paper is currently under preparation.

.. toctree::
   :titlesonly:
   :hidden:

   install
   quickstart
   tutorials
   data
   models
   simulation
   API Documentation <autoapi/easypheno/index>
