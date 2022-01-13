import argparse
import os
import pandas as pd

from utils import helper_functions


def check_and_create_directories(base_dir: str, arguments: argparse.Namespace):
    """
    Function to check if required subdirectories exist at base_dir and to create them if not
    :param base_dir: base directory to check
    :param arguments: arguments handed over by user
    """
    # add all required directories (directories within will be created automatically)
    required_subdirs = [
        'results/' + arguments.genotype_matrix + '/' + arguments.phenotype_matrix + '/' + arguments.phenotype
    ]
    if arguments.model == 'all':
        implemented_models = helper_functions.get_list_of_implemented_models()
        for model_name in implemented_models:
            required_subdirs.append(required_subdirs[0] + '/' + model_name)
    else:
        required_subdirs[0] += '/' + arguments.model
    for subdir in required_subdirs:
        if not os.path.exists(base_dir + subdir):
            os.makedirs(base_dir + subdir)
            print('Created folder ' + base_dir + subdir)


def check_all_specified_arguments(base_dir: str, arguments: argparse.Namespace):
    """
    Function to check all specified arguments for plausibility
    :param base_dir: base directory containing files
    :param arguments: namespace with all arguments
    """
    # Check existence of genotype and phenotype file
    if not os.path.isfile(base_dir + 'data/' + arguments.genotype_matrix):
        raise Exception('Specified genotype file ' + arguments.genotype_matrix + ' does not exist in '
                        + base_dir + 'data/. Please check spelling.')
    if not os.path.isfile(base_dir + 'data/' + arguments.phenotype_matrix):
        raise Exception('Specified phenotype file ' + arguments.phenotype_matrix + ' does not exist in '
                        + base_dir + 'data/. Please check spelling.')
    # Check existence of specified phenotype in phenotype file
    phenotype_file = pd.read_csv(base_dir + 'data/' + arguments.phenotype_matrix)
    if arguments.phenotype not in phenotype_file.columns:
        raise Exception('Specified phenotype ' + arguments.phenotype + ' does not exist in phenotype file '
                        + base_dir + 'data/' + arguments.phenotype_matrix + '. Check spelling.')

    # Check meaningfulness of specified values
    if not(0 <= arguments.maf_percentage <= 20):
        raise Exception('Specified maf value of ' + str(arguments.maf) + ' is invalid, has to be between 0 and 20.')
    if arguments.n_trials < 10:  # TODO: Sinnvoll?
        raise Exception('Specified number of trials with ' + str(arguments.n_trials) + ' is invalid, at least 10.')

    # Check spelling of datasplit and model
    if arguments.datasplit not in ['nested_cv', 'cv-test', 'train-val-test']:
        raise Exception('Specified datasplit ' + arguments.datasplit + ' is invalid, '
                        'has to be: nested_cv | cv-test | train-val-test')
    if (arguments.model != 'all') and (arguments.model not in helper_functions.get_list_of_implemented_models()):
        raise Exception('Specified model "' + arguments.model + '" not found in implemented models nor "all" specified.'
                        + ' Check spelling or if implementation exists. Implemented models: '
                        + ''.join(helper_functions.get_list_of_implemented_models()))
