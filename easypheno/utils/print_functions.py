import numpy as np
import pandas as pd

from ..preprocess import base_dataset
from . import helper_functions


def print_config_info(arguments: dict, dataset: base_dataset.Dataset, task: str):
    """
    Print info of current configuration.

    :param arguments: arguments specified by the user
    :param dataset: dataset used for optimization
    :param task: task that was detected
    """
    print('+++++++++++ CONFIG INFORMATION +++++++++++')
    print('Genotype Matrix: ' + arguments["genotype_matrix"])
    print('Phenotype Matrix: ' + arguments["phenotype_matrix"])
    print('Phenotype: ' + arguments["phenotype"])
    if arguments["encoding"] is not None:
        print('Encoding: ' + arguments["encoding"])
    if arguments["models"] != 'all':
        print('Models: ' + ', '.join(arguments["models"]))
    else:
        print('Models: ' + ', '.join(helper_functions.get_list_of_implemented_models()))
    print('Optuna Trials: ' + str(arguments["n_trials"]))
    if arguments["datasplit"] == 'train-val-test':
        datasplit_params = [arguments["val_set_size_percentage"], arguments["test_set_size_percentage"]]
    elif arguments["datasplit"] == 'cv-test':
        datasplit_params = [arguments["n_innerfolds"], arguments["test_set_size_percentage"]]
    elif arguments["datasplit"] == 'nested-cv':
        datasplit_params = [arguments["n_outerfolds"], arguments["n_innerfolds"]]
    print('Datasplit: ' + arguments["datasplit"] +
          ' (' + helper_functions.get_subpath_for_datasplit(datasplit=arguments["datasplit"],
                                                            datasplit_params=datasplit_params) + ')')
    print('MAF: ' + str(arguments["maf_percentage"]))
    print('Dataset Infos')
    print('- Task detected: ' + task)
    print('- No. of samples: ' + str(dataset.X_full.shape[0]) + ', No. of features: ' + str(dataset.X_full.shape[1]))
    print('- Encoding: ' + str(dataset.encoding))
    if task == 'classification':
        print('- Samples per class: ' + str(np.unique(dataset.y_full, return_counts=True)[1]))
    else:
        print('- Target variable statistics: ')
        print(pd.DataFrame(dataset.y_full).describe()[0])
    print('++++++++++++++++++++++++++++++++++++++++++++')
