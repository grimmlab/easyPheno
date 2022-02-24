import argparse
import os
import inspect
import importlib
import torch
import random
import numpy as np


def get_list_of_implemented_models():
    """
    Create a list of all implemented models based on files existing in 'model' subdirectory of the repository
    """
    # Assumption: naming of python source file is the same as the model name specified by the user
    try:
        model_src_files = os.listdir('../model')
    except:
        model_src_files = os.listdir('model')
    model_src_files.remove('__init__.py')
    model_src_files.remove('base_model.py')
    model_src_files.remove('sklearn_model.py')
    model_src_files.remove('torch_model.py')
    return [model[:-3] for model in model_src_files]


def test_likely_categorical(vector_to_test: list, threshold: float = 0.1):
    """
    Test whether a vector is most likely categorical.
    Simple heuristic: checking if the ratio of unique values in the vector is below a specified threshold
    :param vector_to_test: vector that is tested if it is most likely categorical
    :param threshold: threshold of unique values' ratio to declare vector categorical
    :return: True if the vector is most likely categorical, False otherwise
    """
    return np.unique(vector_to_test).shape[0] / vector_to_test.shape[0] <= threshold


def get_mapping_name_to_class():
    """
    Get a mapping from model name (naming in package model without .py) to class name
    :return: dictionary with mapping model name to class name
    """
    try:
        files = os.listdir('../model')
    except:
        files = os.listdir('model')
    modules_mapped = {}
    for file in files:
        if file not in ['__init__.py', '__pycache__']:
            if file[-3:] != '.py':
                continue

            file_name = file[:-3]
            module_name = 'model.' + file_name
            for name, cls in inspect.getmembers(importlib.import_module(module_name), inspect.isclass):
                if cls.__module__ == module_name:
                    modules_mapped[file_name] = cls
    return modules_mapped


def set_all_seeds(seed: int = 42):
    """
    Set all seeds of libs with a specific function for reproducability of results
    :param seed: seed to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_subpath_for_datasplit(arguments: argparse, datasplit: str) -> str:
    """
    Method to construct the subpath according to the datasplit
    :param arguments: arguments specified by the user
    :param datasplit: datasplit to retrieve
    :return: string with the subpath
    """
    # construct subpath due to the specified datasplit
    if datasplit == 'train-val-test':
        datasplit_string = f'({100 - arguments.test_set_size_percentage}-{arguments.validation_set_size_percentage})-' \
                           f'{arguments.test_set_size_percentage}'
    elif datasplit == 'cv-test':
        datasplit_string = f'{arguments.n_innerfolds}-{arguments.test_set_size_percentage}'
    elif datasplit == 'nested-cv':
        datasplit_string = f'{arguments.n_outerfolds}-{arguments.n_innerfolds}'

    return datasplit_string
