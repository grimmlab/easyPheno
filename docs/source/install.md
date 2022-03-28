### Requirements
We recommend a workflow using [Docker](https://www.docker.com/) to ensure a stable working environment.
Subsequently, we describe the setup and operation according to it. 
If you want to follow our recommendation, **Docker>= 20.10.12** needs to be installed and running on your machine. We provide a Dockerfile as described below.

If you want to use GPU support, you need to install [nvidia-docker-2](https://github.com/NVIDIA/nvidia-docker) and a version of **CUDA>=11.2**.

As an alternative, you can run all programs directly on your machine. 
The pipeline was developed and tested with Python 3.8 and Ubuntu 20.04.
All used Python packages and versions are specified in `Docker/requirements.txt`.

### Setup and Operation
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
    Mount the directory where the repository is placed on your machine, the directory where your phenotype and genotype data is stored and the directory where you want to save your results using the option `-v`.
    Restrict the number of cpus using the option `cpuset-cpus CPU_INDEX_START-CPU_INDEX_STOP` and specify a gpu device using `--gpus device=DEVICE_NUMBER`if you want to use GPU support.
6. In the Docker container, navigate to the to top level of the repository in the mounted directory
   ```bash
    cd /REPO_DIRECTORY/IN/CONTAINER/phenotypeprediction
   ```