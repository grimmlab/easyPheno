# phenotype-pred-framework

[![Python 3.8](https://img.shields.io/badge/Python-3.8-3776AB)](https://www.python.org/downloads/release/python-388/)

## Requirements
We recommend a workflow using [Docker](https://www.docker.com/) to ensure a stable working environment.
Subsequently, we describe the setup and operation according to it. 
If you want to follow our recommendation, **Docker>= 20.10.12** needs to be installed and running on your machine. We provide a Dockerfile as described below.

If you want to use GPU support, you need to install [nvidia-docker-2](https://github.com/NVIDIA/nvidia-docker) and a version of **CUDA>=11.2**.

As an alternative, you can run all programs directly on your machine. 
The pipeline was developed and tested with Python 3.8 and Ubuntu 20.04.
All used Python packages and versions are specified in `Docker/requirements.txt`.

## Setup and Operation
1. Open a Terminal and navigate to the directory where you want to set up the project
2. Clone this repository
    ```bash
    git clone https://github.com/grimmlab/phenotypeprediction.git
    ```
3. Navigate to `Docker` after cloning the repository
   ```bash
    cd phenotypeprediction/Docker
   ```
4. Build a Docker image using the provided Dockerfile tagged with the IMAGENAME of your choice
    ```bash
    docker build -t IMAGENAME .
    ```
5. Run an interactive Docker container based on the created image with a CONTAINERNAME of your choice
    ```bash
    docker run -it -v PATH/TO/REPO/FOLDER:/REPO_DIRECTORY/IN/CONTAINER -v /PATH/TO/DATA/DIRECTORY:/DATA_DIRECTORY/IN/CONTAINER -v /PATH/TO/RESULTS/SAVE/DIRECTORY:/SAVE_DIRECTORY/IN/CONTAINER --cpuset-cpus CPU_INDEX_START-CPU_INDEX_STOP --gpus device=DEVICE_NUMBER --name CONTAINERNAME IMAGENAME
    ```
    Mount the directory where the repository is placed on your machine, the directory where your phenotype and genotype data is stored and the directroy where you want to save your results in the Docker container using the option `-v`.
    Restrict the number of cpus using the option `cpuset-cpus CPU_INDEX_START-CPU_INDEX_STOP` and specify a gpus device using `--gpus device=DEVICE_NUMBER`if you want to use GPU support.
6. In the Docker container, navigate to the to top level of the repository in the mounted directory
   ```bash
    cd /REPO_DIRECTORY/IN/CONTAINER/phenotypeprediction
   ```
   
### Run model optimization
We provide a framework for Bayesian optimization of model hyperparameters using [optuna](https://optuna.readthedocs.io/en/stable/). 
The optimization is started via the command line and one can specify various parameters. One gets an overview of these parameters using the following command:

    python3 run_phenotype_pred_framework.py -h

To run the optimization, one has to specifiy a data directory, a save directory, a genotype and phenotype matrix, a phenotype to be predicted as well as the models that should be optimized (one can specifiy multiple models). 
For example, if you want to run an optimization using the dummy data we provide, use the following command:

    python3 run_phenotype_pred_framework.py --data_dir /REPO_DIRECTORY/IN/CONTAINER/phenotypeprediction/data/test/ --genotype_matrix x_matrix.h5 --phenotype_matrix y_matrix.csv --phenotype continuous_values --models xgboost randomforest cnn

By doing so, the whole optimization pipeline is started for the specified models. One should see command line outputs for the data prepration, a config overview and then the output of the optimization trials. Finally, in `--save_dir`, there will be a subfolder `results` that contains the optimization results in a structure according to the specified optimization parameters.
A list of all available models can be found under `--models` using the above described help command. Beyond that, one can specify various further parameters, also run the help command to check.

## Contributors
This pipeline is developed and maintened by members of the [Bioinformatics](https://bit.cs.tum.de) lab of [Prof. Dr. Dominik Grimm](https://bit.cs.tum.de/team/dominik-grimm/):
- [Florian Haselbeck, M.Sc.](https://bit.cs.tum.de/team/florian-haselbeck/)
- [Maura John, M.Sc.](https://bit.cs.tum.de/team/maura-john/)

## Citation
When using this workflow, please cite our publication:

tbd.
