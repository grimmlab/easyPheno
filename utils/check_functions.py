import os
import pandas as pd

from utils import helper_functions


def check_all_specified_arguments(arguments: dict):
    """
    Check all specified arguments for plausibility
    :param arguments: all arguments provided by the user
    """
    # Check existence of genotype and phenotype file
    if not os.path.isfile(arguments["data_dir"] + '/' + arguments["genotype_matrix"]):
        raise Exception('Specified genotype file ' + arguments["genotype_matrix"] + ' does not exist in '
                        + arguments["data_dir"] + '/. Please check spelling.')
    if not os.path.isfile(arguments["data_dir"] + '/' + arguments["phenotype_matrix"]):
        raise Exception('Specified phenotype file ' + arguments["phenotype_matrix"] + ' does not exist in '
                        + arguments["data_dir"] + '/. Please check spelling.')
    # Check existence of specified phenotype in phenotype file
    phenotype_file = pd.read_csv(arguments["data_dir"] + '/' + arguments["phenotype_matrix"])
    if arguments["phenotype"] not in phenotype_file.columns:
        raise Exception('Specified phenotype ' + arguments["phenotype"] + ' does not exist in phenotype file '
                        + arguments["data_dir"] + '/' + arguments["phenotype_matrix"] + '. Check spelling.')

    # Check meaningfulness of specified values
    if not (0 <= arguments["maf_percentage"] <= 20):
        raise Exception('Specified maf value of ' + str(arguments["maf_percentage"]) 
                        + ' is invalid, has to be between 0 and 20.')
    if not (5 <= arguments["test_set_size_percentage"] <= 30):
        raise Exception('Specified test set size in percentage ' + str(arguments["test_set_size_percentage"]) +
                        ' is invalid, has to be between 5 and 30.')
    if not (5 <= arguments["val_set_size_percentage"] <= 30):
        raise Exception('Specified validation set size in percentage ' 
                        + str(arguments["val_set_size_percentage"]) + ' is invalid, has to be between 5 and 30.')
    if not (3 <= arguments["n_outerfolds"] <= 10):
        raise Exception('Specified number of outerfolds ' + str(arguments["n_outerfolds"]) +
                        ' is invalid, has to be between 3 and 10.')
    if not (3 <= arguments["n_innerfolds"] <= 10):
        raise Exception('Specified number of innerfolds/folds ' + str(arguments["n_innerfolds"]) +
                        ' is invalid, has to be between 3 and 10.')
    if arguments["n_trials"] < 10:
        raise Exception('Specified number of trials with ' + str(arguments["n_trials"]) + ' is invalid, at least 10.')

    # Check spelling of datasplit and model
    if arguments["datasplit"] not in ['nested-cv', 'cv-test', 'train-val-test']:
        raise Exception('Specified datasplit ' + arguments["datasplit"] + ' is invalid, '
                        'has to be: nested-cv | cv-test | train-val-test')
    if (arguments["models"] != 'all') and \
            (any(model not in helper_functions.get_list_of_implemented_models() for model in arguments["models"])):
        raise Exception('At least one specified model in "' + str(arguments["models"]) +
                        '" not found in implemented models nor "all" specified.' +
                        ' Check spelling or if implementation exists. Implemented models: ' +
                        str(helper_functions.get_list_of_implemented_models()))

    # Check encoding
    if arguments["encoding"] is not None:
        if arguments["encoding"] not in ['raw', '012', 'onehot']:
            raise Exception('Specified encoding ' + arguments["encoding"] + ' is not valid. See help.')
        else:
            if arguments["models"] == 'all' or len(arguments["models"]) > 1:
                raise Exception('If "all" models are specified, standard encodings are used. Do not specify encoding')
            else:
                if arguments["encoding"] not in \
                        helper_functions.get_mapping_name_to_class()[arguments["models"][0]].possible_encodings:
                    raise Exception(arguments["encoding"] + ' is not valid for ' + arguments["models"][0] +
                                    '. Check possible_encodings in model file.')

    # Only relevant for neural networks
    if arguments["batch_size"] is not None:
        if not (2**3 <= arguments["batch_size"] <= 2**8):
            raise Exception('Specified batch size ' + str(arguments["batch_size"]) +
                            ' is invalid, has to be between 8 and 256.')
    if arguments["n_epochs"] is not None:
        if not (50 <= arguments["n_epochs"] <= 1000000):
            raise Exception('Specified number of epochs ' + str(arguments["n_epochs"]) +
                            ' is invalid, has to be between 50 and 1.000.000.')
