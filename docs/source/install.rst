Installation Guide
===================
easyPheno offers two ways of using it:

- :ref:`Docker Workflow`: run the optimization pipeline with only one command in a Docker container
- :ref:`pip Workflow`: use our framework as a Python library and integrate it into your pipeline

The whole framework was developed and tested using `Ubuntu 20.04 <https://releases.ubuntu.com/20.04/>`_. Consequently,
the subsequent guide is mainly written with regard to `Ubuntu 20.04 <https://releases.ubuntu.com/20.04/>`_.
The framework should work with Windows and Mac as well, but we do not officially provide support for these platforms.

Besides the written guides, we also provide some tutorial videos which are embedded below.

Docker Workflow
-----------------------
If you want to do phenotype prediction without the need of integrating parts of your own pipeline,
we recommend the :ref:`Docker Workflow`: due to its easy-to-use interface and ready-to-use working environment
within a `Docker <https://www.docker.com/>`_ container.

Requirements
~~~~~~~~~~~~~~~~~~~~~~
For the :ref:`Docker Workflow`, `Docker <https://www.docker.com/>`_ needs to be installed and running on your machine,
see the `Installation Guidelines at the Docker website <https://docs.docker.com/get-docker/>`_.
On Ubuntu, you can use ``docker run hello-world`` to check if Docker works
(Caution: add sudo if you are not in the docker group).

If you want to use GPU support, you need to install `nvidia-docker-2 <https://github.com/NVIDIA/nvidia-docker>`_ (see this `nvidia-docker Installation Guide <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit>`_)
and a version of `CUDA <https://developer.nvidia.com/cuda-toolkit>`_ >= 11.2 (see this `CUDA Installation Guide <https://docs.nvidia.com/cuda/index.html#installation-guides>`_). To check your CUDA version, just run ``nvidia-smi`` in a terminal.

Setup
~~~~~~~~~~~~~~~~~~~~~~
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


Your setup is finished! Go to :ref:`HowTo: Run easyPheno using Docker` to see how you can now use easyPheno!

Useful Docker commands
~~~~~~~~~~~~~~~~~~~~~~
The subsequent Docker commands might be useful when using easyPheno.
See `here <https://docs.docker.com/engine/reference/commandline/docker/>`_ for a full guide on the Docker commands.

:docker images: List all Docker images on your machine
:docker ps: List all running Docker containers on your machine
:docker ps -a: List all Docker containers (including stopped ones) on your machine
:docker start -i CONTAINERNAME: Start a (stopped) Docker container interactively to enter its command line interface


pip Workflow
-----------------------
easyPheno can be installed via ``pip`` and used as a common Python library.


The pipeline was developed and tested with `Python 3.8 <https://www.python.org/downloads/release/python-3813/>`_ and `Ubuntu 20.04 <https://releases.ubuntu.com/20.04/>`_.
The framework should work with Windows and Mac as well, but we do not officially provide support for these platform.
We neither officially support other Python versions, however easyPheno might run as well.


Just install easyPheno via ``pip``:

    .. code-block::

        pip install PACKAGENAME

Our setup is finished! Go to :ref:`HowTo: Run easyPheno as a pip package` to see how you can now use easyPheno!

