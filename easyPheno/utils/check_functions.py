import pathlib
import pandas as pd
import numpy as np

from . import helper_functions
from ..model import _param_free_base_model, _torch_model, _tensorflow_model


def check_all_specified_arguments(arguments: dict):
    """
    Check all specified arguments for plausibility

    :param arguments: all arguments provided by the user
    """
    # Check existence of save_dir
    if not arguments["save_dir"].exists():
        raise Exception("Specified save_dir " + str(arguments["save_dir"]) + " does not exist. Please double-check.")
    # Check existence of genotype and phenotype file
    if not arguments["data_dir"].joinpath(arguments["genotype_matrix"]).is_file():
        raise Exception('Specified genotype file ' + arguments["genotype_matrix"] + ' does not exist in '
                        + str(arguments["data_dir"]) + '. Please check spelling.')
    phenotype_file = arguments["data_dir"].joinpath(arguments["phenotype_matrix"])
    if not phenotype_file.is_file():
        raise Exception('Specified phenotype file ' + arguments["phenotype_matrix"] + ' does not exist in '
                        + str(arguments["data_dir"]) + '. Please check spelling.')
    # Check existence of specified phenotype in phenotype file
    phenotype = pd.read_csv(phenotype_file)
    if arguments["phenotype"] not in phenotype.columns:
        raise Exception('Specified phenotype ' + arguments["phenotype"] + ' does not exist in phenotype file '
                        + str(phenotype_file) + '. Check spelling.')

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
    if "n_trials" in arguments and any([not issubclass(helper_functions.get_mapping_name_to_class()[model],
                                                       _param_free_base_model.ParamFreeBaseModel)
                                        for model in arguments["models"]]) and arguments["n_trials"] < 10:
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
    if any([issubclass(helper_functions.get_mapping_name_to_class()[model], _torch_model.TorchModel) or \
            issubclass(helper_functions.get_mapping_name_to_class()[model] ,  _tensorflow_model.TensorflowModel)
            for model in arguments["models"]]):
        if arguments["batch_size"] is not None:
            if not (2**3 <= arguments["batch_size"] <= 2**8):
                raise Exception('Specified batch size ' + str(arguments["batch_size"]) +
                                ' is invalid, has to be between 8 and 256.')
        if arguments["n_epochs"] is not None:
            if not (50 <= arguments["n_epochs"] <= 1000000):
                raise Exception('Specified number of epochs ' + str(arguments["n_epochs"]) +
                                ' is invalid, has to be between 50 and 1.000.000.')


def check_exist_directories(list_of_dirs: list, create_if_not_exist: bool = False) -> bool:
    """
    Check if each directory within a list exists

    :param list_of_dirs: list with directories as pathlib.Path
    :param create_if_not_exist: bool if non-existing directories should be created

    :return: True if all exist, False otherwise
    """
    check = True
    for dir_to_check in list_of_dirs:
        if not dir_to_check.exists():
            print("Directory " + str(dir_to_check) + " not existing.")
            if create_if_not_exist:
                print("Will create it.")
                dir_to_check.mkdir(parents=True)
            else:
                print("Please correct it.")
                check = False
    return check


def check_exist_files(list_of_files: list) -> bool:
    """
    Check if each file within a list exists

    :param list_of_files: list with files as pathlib.Path

    :return: True if all exist, False otherwise
    """
    check = True
    for file_to_check in list_of_files:
        if not file_to_check.is_file():
            print("File " + str(file_to_check) + " not existing.")
            print("Please correct it.")
            check = False
    return check


def compare_snp_id_vectors(snp_id_vector_small_equal: np.array, snp_id_vector_big_equal: np.array) -> bool:
    """
    Compare two SNP id vectors if they contain the same ids

    :param snp_id_vector_small_equal: vector 1 with SNP ids, can be a (smaller) subset of snp_id_vector_big_equal
    :param snp_id_vector_big_equal: vector 2 with SNP ids, can contain more SNP ids than snp_id_vector_small_equal

    :return: True if snp_id_vector_small_equal is a subset of the other vector
    """
    return set(snp_id_vector_small_equal).issubset(set(snp_id_vector_big_equal))
