import argparse
import os

from utils import helper_functions


def check_and_create_directories(base_dir: str, arguments: argparse.Namespace):
    """
    Function to check if required subdirectories exist at base_dir and to create them if not
    :param base_dir: base directory to check
    :param arguments: arguments handed over by user
    """
    # add all required directories (directories in between will be created automatically)
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


def check_all_specified_arguments(arguments: argparse.Namespace):
    """
    Function to check all specified arguments for plausibility
    :param arguments: namespace with all arguments
    """
    # TODO: implement check of all specified args
    # prüfen ob werte passend sind
    # prüfen ob dateien existieren
    # prüfen ob phenotype vorhanden ist
    # wenn nicht, dann exception und abbruch

    if (arguments.model != 'all') and (arguments.model not in helper_functions.get_list_of_implemented_models()):
        raise Exception('Specified model "' + arguments.model + '" not found in implemented models nor "all" specified.'
                        + ' Check spelling or if implementation exists. Implemented models: '
                        + ''.join(helper_functions.get_list_of_implemented_models()))
