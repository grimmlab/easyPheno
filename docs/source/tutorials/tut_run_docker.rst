HowTo: Run easyPheno using Docker
======================================
We assume that you succesfully did all steps described in :ref:`Docker workflow`: to setup easyPheno using Docker.
Besides this written tutorial, we recorded a :ref:`Video tutorial: Run easyPheno with Docker` embedded below.

Workflow
"""""""""""
You are at the **root directory within your Docker container**, i.e. after step 5 of the setup at :ref:`Docker workflow`:.

If you closed the Docker container you created at the end of the installation, just use ``docker start -i CONTAINERNAME``
to start it in interactive mode again. If you did not create a container yet, go back to step 5 of the setup.

1. Navigate to the directory where the easyPheno repository is placed within your container

    .. code-block::

        cd /REPO_DIRECTORY/IN/CONTAINER/easyPheno

2. Run easyPheno (as module). By default, easyPheno starts the optimization procedure for 10 trials with XGBoost and a 5-fold nested cross-validation using the data we provide in ``tutorials/tutorial_data``.

    .. code-block::

        python3 -m easypheno.run --save_dir SAVE_DIRECTORY

    That's it! Very easy! You can now find the results in the save directory you specified.

3. To get an overview of the different options you can set for running easyPheno, just do:

    .. code-block::

        python3 -m easypheno.run --help


Feel free to test easyPheno, e.g. with other prediction models.
If you want to start using your own data, please carefully read our :ref:`Data Guide`: to ensure that your data fulfills all requirements.

Video tutorial: Run easyPheno with Docker
""""""""""""""""""""""""""""""""""""""""""""""""""
.. youtube:: uM-yJqzzIPo
    :width: 640
    :height: 360
    :aspect: 16:9