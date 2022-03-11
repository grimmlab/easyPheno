import argparse
import numpy as np
import pandas as pd

from preprocess import base_dataset
from utils import helper_functions


def print_config_info(arguments: argparse.Namespace, dataset: base_dataset.Dataset, task: str):
    """
    Function to print info of current configuration.
    :param arguments: arguments specified by the user
    :param dataset: dataset used for optimization
    :param task: task that was detected
    """
    # TODO: implement function
    # config infos, nmber of samples etc., welches Modell gewählt wurde, data split etc.
    # wichtige infos einfach auf der command line ausgeben
    print('+++++++++++ CONFIG INFORMATION +++++++++++')
    print('Genotype Matrix: ' + arguments.genotype_matrix)
    print('Phenotype Matrix: ' + arguments.phenotype_matrix)
    print('Phenotype: ' + arguments.phenotype)
    if arguments.encoding is not None:
        print('Encoding: ' + arguments.encoding)
    print('Models: ' + ', '.join(arguments.models))
    print('Optuna Trials: ' + str(arguments.n_trials))
    print('Datasplit: ' + arguments.datasplit +
          ' (' + helper_functions.get_subpath_for_datasplit(arguments=arguments, datasplit=arguments.datasplit) + ')')
    print('MAF: '+ str(arguments.maf_percentage))
    print('Dataset Infos')
    print('- Task detected: ' + task)
    print('- No. of samples: ' + str(dataset.X_full.shape[0]) + ', No. of features: ' + str(dataset.X_full.shape[1]))
    if task == 'classification':
        print('- Samples per class: ' + str(np.unique(dataset.y_full, return_counts=True)[1]))
    else:
        print('- Target variable statistics: ')
        print(pd.DataFrame(dataset.y_full).describe()[0])
    print('++++++++++++++++++++++++++++++++++++++++++++')
