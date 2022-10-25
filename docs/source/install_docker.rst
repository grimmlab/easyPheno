Docker workflow
======================
If you want to do phenotype prediction without the need of integrating parts of your own pipeline,
we recommend the :ref:`Docker Workflow` due to its easy-to-use interface and ready-to-use working environment
within a `Docker <https://www.docker.com/>`_ container. Besides the written tutorial, we provide a :ref:`Video tutorial: Docker workflow setup` embedded below.

Requirements
"""""""""""""""""""""""""""""""""""""""""""""""
For the :ref:`Docker Workflow`, `Docker <https://www.docker.com/>`_ needs to be installed and running on your machine,
see the `Installation Guidelines at the Docker website <https://docs.docker.com/get-docker/>`_.
On Ubuntu, you can use ``docker run hello-world`` or ```docker --version`` to check if Docker works
(Caution: add sudo if you are not in the docker group).

If you want to use GPU support, you need to install `nvidia-docker-2 <https://github.com/NVIDIA/nvidia-docker>`_ (see this `nvidia-docker Installation Guide <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit>`_)
and a version of `CUDA <https://developer.nvidia.com/cuda-toolkit>`_ >= 11.2 (see this `CUDA Installation Guide <https://docs.nvidia.com/cuda/index.html#installation-guides>`_). To check your CUDA version, just run ``nvidia-smi`` in a terminal.

Setup
"""""""""""""""""""""""""""""""""""""""""""""""
1. Open a Terminal and navigate to the directory where you want to set up the project
2. Clone this repository

    .. code-block::

        git clone https://github.com/grimmlab/easyPheno.git

3. Navigate to `Docker` after cloning the repository

    .. code-block::

        cd easyPheno/Docker

4. Build a Docker image using the provided Dockerfile tagged with the IMAGENAME of your choice

    .. code-block::

        docker build -t IMAGENAME .

5. Run an interactive Docker container based on the created image with a CONTAINERNAME of your choice

    .. code-block::

        docker run -it -v /PATH/TO/REPO/FOLDER:/REPO_DIRECTORY/IN/CONTAINER -v /PATH/TO/DATA/DIRECTORY:/DATA_DIRECTORY/IN/CONTAINER -v /PATH/TO/RESULTS/SAVE/DIRECTORY:/SAVE_DIRECTORY/IN/CONTAINER --name CONTAINERNAME IMAGENAME

    - Mount the directory where the repository is placed on your machine, the directory where your phenotype and genotype data is stored and the directory where you want to save your results using the option ``-v``.
    - You can restrict the number of cpus using the option ``cpuset-cpus CPU_INDEX_START-CPU_INDEX_STOP``.
    - Specify a gpu device using ``--gpus device=DEVICE_NUMBER`` if you want to use GPU support.


    Let's have a look at an example. We assume hat you created a Docker image called ``ep-image``, your repository and data is placed in (subfolders of) ``/myhome/``, you want to save your results to ``/myhome/`` (so ``/myhome/`` is the only directory you need to mount in your container), you only want to use CPUs 0 to 9 and GPU 0 and you want to call your container ``ep_container``. Then you have to run the following command:

    .. code-block::

        docker run -it -v /myhome/:/myhome_in_my_container/ --cpuset-cpus 0-9 --gpus device=0 --name ep_container ep_image

Your setup is finished! Go to :ref:`HowTo: Run easyPheno using Docker` to see how you can now use easyPheno!

Useful Docker commands
"""""""""""""""""""""""""""""""""""""""""""""""
The subsequent Docker commands might be useful when using easyPheno.
See `here <https://docs.docker.com/engine/reference/commandline/docker/>`_ for a full guide on the Docker commands.

:docker images: List all Docker images on your machine
:docker ps: List all running Docker containers on your machine
:docker ps -a: List all Docker containers (including stopped ones) on your machine
:docker start -i CONTAINERNAME: Start a (stopped) Docker container interactively to enter its command line interface

Video tutorial: Docker workflow setup
"""""""""""""""""""""""""""""""""""""""""""""
.. youtube:: ONC9MZic0n0
    :width: 640
    :height: 360
    :aspect: 16:9
