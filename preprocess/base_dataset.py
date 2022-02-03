import argparse
import h5py
import pandas as pd
import numpy as np
from utils import helper_functions


class Dataset:
    """Class containing dataset ready for optimization (e.g. geno/phenotype matched)"""

    def __init__(self, arguments: argparse.Namespace, encoding: str):
        self.encoding = encoding
        self.X_full, self.y_full = self.load_match_raw_data(arguments=arguments)
        self.maf_filter_raw_data(arguments=arguments)
        self.datasplit = arguments.datasplit
        self.datasplit_indices = self.load_datasplit_indices(arguments=arguments)

    def load_match_raw_data(self, arguments: argparse.Namespace):
        """
        Load the full genotype and phenotype matrices specified and match them
        :param arguments: all arguments provided by the user
        """
        # TODO: MAURA - Rohdaten in richtiger Codierung laden
        with h5py.File(arguments.base_dir + '/data/' + arguments.genotype_matrix.split('.')[0] + '.h5', "r") as f:
            X_full_raw = f[f'X_full_{self.encoding}'][
                         :]  # TODO: @Maura: bitte "Pfad in .h5" und Datei laden nochmal prüfen
        phenotype_matrix = \
            pd.read_csv(arguments.base_dir + '/data/' + arguments.phenotype_matrix.split('.')[0] + '.csv')
        # TODO: @Maura: das laden von der .csv hab ich schon mal generisch gemacht, falls über phenotype_matrix mal ein anderes Format übergeben werden kann
        # Please do the matching magic and return matched X_full and y_full ;-) Sinnvoll wäre da halt wieder eine Funktion in raw_data_functions, weil wir das zum Ertsellen der datasplits ja auch schon machen (und dann halt nicht speichern)
        return ...

    def maf_filter_raw_data(self, arguments: argparse.Namespace):
        """
        Apply maf filter to full raw data
        :param arguments: all arguments provided by the user
        """
        with h5py.File(arguments.base_dir + '/data/' + self.get_index_file_name(), "r") as f:
            if f'maf_filter/maf_{arguments.maf_percentage}' in f:
                filter_indices = f[f'maf_filter/maf_{arguments.maf_percentage}'][:]
                # TODO: @MAURA Bitte Pfad prüfen! percentage ohne führende 0 (z. B. bei 1 Prozent) vmtl. einfacher)
                # "nicht-standard-maf-filter" sollten schon vorher angelegt werden, wenn wir die "index_files" bauen bzw. prüfen
        self.X_full = np.delete(self.X_full, filter_indices, axis=1)

    def load_datasplit_indices(self, arguments: argparse.Namespace):
        """
        Load the datasplit indices saved during file unification
        Structure: {
                    'outerfold_0':
                        {
                        'innerfold_0': {'train': indices_train, 'val': indices_val},
                        'innerfold_1': {'train': indices_train, 'val': indices_val},
                        ...
                        'innerfold_n': {'train': indices_train, 'val': indices_val},
                        'test': test_indices
                        },
                    ...
                    'outerfold_m':
                         {
                        'innerfold_0': {'train': indices_train, 'val': indices_val},
                        'innerfold_1': {'train': indices_train, 'val': indices_val},
                        ...
                        'innerfold_n': {'train': indices_train, 'val': indices_val},
                        'test': test_indices
                        }
                    }
        Caution: The actual structure depends on the datasplit specified by the user,
        e.g. for a train-val-test split only 'outerfold_0' and its subelements 'innerfold_0' and 'test' exist.
        :param arguments: all arguments provided by the user
        :return: dictionary with the above-described structure containing all indices for the specified data split
        """
        # TODO: @MAURA: Bitte überprüf die Logik mal ob das so zur Dateistruktur passt
        if self.datasplit == 'train-val-test':
            n_outerfolds = 1
            n_innerfolds = 1
        elif self.datasplit == 'cv-test':
            n_outerfolds = 1
            n_innerfolds = arguments.n_innerfolds
        elif self.datasplit == 'nested-cv':
            n_outerfolds = arguments.n_outerfolds
            n_innerfolds = arguments.n_innerfolds
        split_param_string = helper_functions.get_subpath_for_datasplit(arguments=arguments)

        datasplit_indices = {}
        with h5py.File(arguments.base_dir + '/data/' + self.get_index_file_name(), "r") as f:
            for m in range(n_outerfolds):
                outerfold_path = \
                    f'datasplits/{self.datasplit}/{split_param_string}/outerfold_{m}/'
                datasplit_indices['outerfold_' + str(m)] = {'test': f[f'{outerfold_path}test/'][:]}
                for n in range(n_innerfolds):
                    datasplit_indices['outerfold_' + str(m)]['innerfold_' + str(n)] = \
                        {
                            'train': f[f'{outerfold_path}innerfold_{n}/train'][:],
                            'val': f[f'{outerfold_path}innerfold_{n}/val'][:]
                        }

        return datasplit_indices

    @staticmethod
    def get_index_file_name(arguments: argparse.Namespace):
        """
        Get the name of the file containing the indices for maf filters and data splits
        :param arguments: all arguments provided by the suer
        :return: name of index file
        """
        return arguments.genotype_matrix.split('.')[0] + '-' + \
            arguments.phenotype_matrix.split('.')[0] + '-' + arguments.phenotype + '.h5'
