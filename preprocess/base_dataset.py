import argparse
import h5py
import numpy as np
import sklearn.preprocessing

from utils import helper_functions
from preprocess import raw_data_functions
from preprocess import encoding_functions


class Dataset:
    """Class containing dataset ready for optimization (e.g. geno/phenotype matched)"""

    def __init__(self, arguments: argparse.Namespace, encoding: str):
        self.encoding = encoding
        self.X_full, self.y_full, self.sample_ids_full = self.load_match_raw_data(arguments=arguments)
        self.maf_filter_raw_data(arguments=arguments)
        self.datasplit = arguments.datasplit
        self.datasplit_indices = self.load_datasplit_indices(arguments=arguments)

    def load_match_raw_data(self, arguments: argparse.Namespace):
        """
        Load the full genotype and phenotype matrices specified and match them
        :param arguments: all arguments provided by the user
        """
        with h5py.File(arguments.base_dir + '/data/' + arguments.genotype_matrix.split('.')[0] + '.h5', "r") as f:
            if f'X_{self.encoding}' in f:
                X = f[f'X_{self.encoding}'][:]
            elif f'X_{encoding_functions.get_base_encoding(self.encoding)}' in f:
                X_base = f[f'X_{encoding_functions.get_base_encoding(self.encoding)}'][:]
                X = encoding_functions.encode_genotype(X_base, self.encoding,
                                                       encoding_functions.get_base_encoding(self.encoding))
            else:
                raise Exception('Genotype in ' + self.encoding + ' encoding missing. Can not create required encoding. '
                                                                 'See documentation for help')

        with h5py.File(arguments.base_dir + '/data/' + self.get_index_file_name(arguments), "r") as f:
            X = raw_data_functions.get_matched_data(X, f['matched_data/X_index'][:])
            y = f['matched_data/y'][:]  # TODO change if multiple phenotypes
            if helper_functions.test_likely_categorical(y):
                if y.dtype.type is np.float64:
                    y = y.astype(int)
                y = sklearn.preprocessing.LabelEncoder().fit_transform(y)
            sample_ids = f['matched_data/matched_sample_ids'][:]
        return X, np.reshape(y, (-1, 1)), np.reshape(sample_ids, (-1, 1))

    def maf_filter_raw_data(self, arguments: argparse.Namespace):
        """
        Apply maf filter to full raw data
        :param arguments: all arguments provided by the user
        """
        with h5py.File(arguments.base_dir + '/data/' + self.get_index_file_name(arguments), "r") as f:
            if f'maf_filter/maf_{arguments.maf_percentage}' in f:
                filter_indices = f[f'maf_filter/maf_{arguments.maf_percentage}'][:]
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
        if self.datasplit == 'train-val-test':
            n_outerfolds = 1
            n_innerfolds = 1
        elif self.datasplit == 'cv-test':
            n_outerfolds = 1
            n_innerfolds = arguments.n_innerfolds
        elif self.datasplit == 'nested-cv':
            n_outerfolds = arguments.n_outerfolds
            n_innerfolds = arguments.n_innerfolds
        split_param_string = helper_functions.get_subpath_for_datasplit(arguments=arguments,
                                                                        datasplit=arguments.datasplit)

        datasplit_indices = {}
        with h5py.File(arguments.base_dir + '/data/' + self.get_index_file_name(arguments), "r") as f:
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
