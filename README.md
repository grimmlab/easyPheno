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
    Mount the directory where the repository is placed on your machine, the directory where your phenotype and genotype data is stored and the directory where you want to save your results using the option `-v`.
    Restrict the number of cpus using the option `cpuset-cpus CPU_INDEX_START-CPU_INDEX_STOP` and specify a gpu device using `--gpus device=DEVICE_NUMBER`if you want to use GPU support.
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

    python3 run_phenotype_pred_framework.py --data_dir /REPO_DIRECTORY/IN/CONTAINER/phenotypeprediction/data/test/ --save_dir /SAVE_DIRECTORY/IN/CONTAINER --genotype_matrix x_matrix.h5 --phenotype_matrix y_matrix.csv --phenotype continuous_values --models xgboost randomforest cnn

By doing so, the whole optimization pipeline is started for the specified models. One should see command line outputs for the data prepration, a config overview and then the output of the optimization trials. Finally, in `--save_dir`, there will be a subfolder `results` that contains the optimization results in a structure according to the specified optimization parameters.
A list of all available models can be found under `--models` using the above described help command. Beyond that, one can specify various further parameters, also run the help command to check.

## Input data
The minimal requirement is to provide a genotype and a phenotype file. We provide test data in the folder `data/test`. The genotype matrix needs to be imputed.
This Framework is designed to work with several genotype file formats. In a first step the genotype data will be saved 
in a unified .h5 file which simplifies further processing. One can provide the following file formats:

### .hdf5/.h5/.h5py
The file has to contain the following keys:

- X_raw: genotype matrix in raw nucleotides (i.e. 'A', 'C', 'G', 'T', 'M', 'R', 'W', 'S', 'Y', 'K')
        with samples as rows and markers as columns
- sample_ids: vector containing corresponding sample ids in the same order as the genotype matrix
- snp_ids: vector containing the identifiers of all SNPs in the same order as the genotype matrix


### .csv
The first column should be the sample ids. The column names should be the SNP identifiers. The values should be the 
genotype matrix in raw encoding (i.e. 'A', 'C', 'G', 'T', 'M', 'R', 'W', 'S', 'Y', 'K'). 


### PLINK
To use PLINK data, a .map and .ped file with the same prefix need to be in the same folder. 
To run the Framework with PLINK files, you can use PREFIX.map or PREFIX.ped as option for the genotype file.


### binary PLINK
To use binary PLINK data, a .bed, .bim and .fam file with the same prefix need to be in the same folder. 
To run the Framework with binary PLINK files, you can use PREFIX.bed, PREFIX.bim or PREFIX.fam as option for the 
genotype file.

### phenotype file 
The Framework currently only accepts .csv, .pheno and .txt files for the phenotype. Here the first column should contain
the sample ids. The remaining columns should contain the phenotype values with the phenotype name as column name. 
For .txt and .pheno files it is assumed that the values are separated by a single space.

## Preprocessing

For each genotype-phenotype combination a separate index file will be created. This file contains the sample indices to 
quickly re-match the genotype and phenotype matrices as well as datasets with indices for different data splits and MAF 
filters. This way the data splits are the same for all models. Additionally, the sample ids and minor allele frequencies 
for all SNPs are stored to easily create new MAF filters and data splits and append to the index file.
When first creating the index file, some standard values for the data splits and MAF filters will be used additionally 
to the values specified by the user. The index file has the following format:
        
    'matched_data': {
        'y': matched phenotypic values,
        'matched_sample_ids': sample ids of matched genotype/phenotype,
        'X_index': indices of genotype matrix to redo matching,
        'y_index': indices of phenotype vector to redo matching,
        'ma_frequency': minor allele frequency of each SNP in genotype file 
        }
    'maf_filter': {
        'maf_{maf_percentage}': indices of SNPs to delete (with MAF < maf_percentage),
        ...
        }
    'datasplits': {
        'nested_cv': {
            '#outerfolds-#innerfolds': {
                'outerfold_0': {
                    'innerfold_0': {'train': indices_train, 'val': indices_val},
                    ...
                    'innerfold_n': {'train': indices_train, 'val': indices_val},
                    'test': test_indices
                    },
                ...
                'outerfold_m': {
                    'innerfold_0': {'train': indices_train, 'val': indices_val},
                    ...
                    'innerfold_n': {'train': indices_train, 'val': indices_val},
                    'test': test_indices
                    }
                },
                ...
            }
        'cv-test': {
            '#folds-test_percentage': {
                'outerfold_0': {
                    'innerfold_0': {'train': indices_train, 'val': indices_val},
                    ...
                    'innerfold_n': {'train': indices_train, 'val': indices_val},
                    'test': test_indices
                    }
                },
                ...
            }
        'train-val-test': {
            'train_percentage-val_percentage-test_percentage': {
                'outerfold_0': {
                    'innerfold_0': {'train': indices_train, 'val': indices_val},
                    'test': test_indices
                    }
                },
                ...
            }
        }

## Contributors
This pipeline is developed and maintained by members of the [Bioinformatics](https://bit.cs.tum.de) lab of [Prof. Dr. Dominik Grimm](https://bit.cs.tum.de/team/dominik-grimm/):
- [Florian Haselbeck, M.Sc.](https://bit.cs.tum.de/team/florian-haselbeck/)
- [Maura John, M.Sc.](https://bit.cs.tum.de/team/maura-john/)

## Citation
When using this workflow, please cite our publication:

tbd.
