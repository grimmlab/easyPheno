import h5py
import numpy as np
import sklearn.preprocessing

from ..utils import helper_functions
from . import raw_data_functions, encoding_functions


class Dataset:
    """
    Class containing dataset ready for optimization (e.g. geno/phenotype matched).

    **Attributes**

        - encoding (*str*): the encoding to use (standard encoding or user-defined)
        - X_full (*numpy.array*): all (matched, maf- and duplicated-filtered) SNPs
        - y_full (*numpy.array*): all target values
        - sample_ids_full (*numpy.array*):all sample ids
        - snp_ids (*numpy.array*): SNP ids
        - datasplit (*str*): datasplit to use
        - datasplit_indices (*dict*): dictionary containing all indices for the specified datasplit

    :param data_dir: data directory where the phenotype and genotype matrix are stored
    :param genotype_matrix_name: name of the genotype matrix including datatype ending
    :param phenotype_matrix_name: name of the phenotype matrix including datatype ending
    :param phenotype: name of the phenotype to predict
    :param datasplit: datasplit to use. Options are: nested-cv, cv-test, train-val-test
    :param n_outerfolds: number of outerfolds relevant for nested-cv
    :param n_innerfolds: number of folds relevant for nested-cv and cv-test
    :param test_set_size_percentage: size of the test set relevant for cv-test and train-val-test
    :param val_set_size_percentage: size of the validation set relevant for train-val-test
    :param maf_percentage: threshold for MAF filter as percentage value
    :param encoding: the encoding to use (standard encoding or user-defined)
    """

    def __init__(self, data_dir: str, genotype_matrix_name: str, phenotype_matrix_name: str, phenotype: str,
                 datasplit: str, n_outerfolds: int, n_innerfolds: int, test_set_size_percentage: int,
                 val_set_size_percentage: int, encoding: str, maf_percentage: int):
        self.encoding = encoding
        self.datasplit = datasplit
        self.index_file_name = self.get_index_file_name(
                genotype_matrix_name=genotype_matrix_name, phenotype_matrix_name=phenotype_matrix_name,
                phenotype=phenotype
        )
        self.X_full, self.y_full, self.sample_ids_full, self.snp_ids = self.load_match_raw_data(
            data_dir=data_dir, genotype_matrix_name=genotype_matrix_name)
        self.maf_filter_raw_data(data_dir=data_dir, maf_percentage=maf_percentage)
        self.filter_duplicate_snps()
        self.datasplit_indices = self.load_datasplit_indices(
            data_dir=data_dir, n_outerfolds=n_outerfolds, n_innerfolds=n_innerfolds,
            test_set_size_percentage=test_set_size_percentage, val_set_size_percentage=val_set_size_percentage
        )

    def load_match_raw_data(self, data_dir: str, genotype_matrix_name: str) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Load the full genotype and phenotype matrices specified and match them

        :param data_dir: data directory where the phenotype and genotype matrix are stored
        :param genotype_matrix_name: name of the genotype matrix including datatype ending

        :return: matched genotype, phenotype and sample ids
        """
        print('Load and match raw data')
        with h5py.File(data_dir + '/' + self.index_file_name, "r") as f:
            # Load information from index file
            y = f['matched_data/y'][:]  # TODO change if multiple phenotypes
            if helper_functions.test_likely_categorical(y):
                if y.dtype.type is np.float64:
                    y = y.astype(int)
                y = sklearn.preprocessing.LabelEncoder().fit_transform(y)
            sample_ids = f['matched_data/matched_sample_ids'][:].astype(str)
            non_informative_filter = f['matched_data/non_informative_filter'][:]
            X_index = f['matched_data/X_index'][:]

        with h5py.File(data_dir + '/' + genotype_matrix_name.split('.')[0] + '.h5', "r") as f:
            # Load genotype data
            snp_ids = f['snp_ids'][non_informative_filter].astype(str)
            if f'X_{self.encoding}' in f:
                X = f[f'X_{self.encoding}'][:, non_informative_filter]
            elif f'X_{encoding_functions.get_base_encoding(encoding=self.encoding)}' in f:
                X_base = \
                    f[f'X_{encoding_functions.get_base_encoding(encoding=self.encoding)}'][:, non_informative_filter]
                X = encoding_functions.encode_genotype(X=X_base, required_encoding=self.encoding)
            else:
                raise Exception('Genotype in ' + self.encoding + ' encoding missing. Can not create required encoding. '
                                                                 'See documentation for help')
        X = raw_data_functions.get_matched_data(data=X, index=X_index)
        return X, np.reshape(y, (-1, 1)), np.reshape(sample_ids, (-1, 1)), snp_ids

    def maf_filter_raw_data(self, data_dir: str, maf_percentage: int):
        """
        Apply maf filter to full raw data, if maf=0 only non-informative SNPs will be removed

        :param data_dir: data directory where the phenotype and genotype matrix are stored
        :param maf_percentage: threshold for MAF filter as percentage value
        """
        print('Apply MAF filter')
        with h5py.File(data_dir + '/' + self.index_file_name, "r") as f:
            if f'maf_filter/maf_{maf_percentage}' in f:
                filter_indices = f[f'maf_filter/maf_{maf_percentage}'][:]
        self.X_full = np.delete(self.X_full, filter_indices, axis=1)
        self.snp_ids = np.delete(self.snp_ids, filter_indices)

    def filter_duplicate_snps(self):
        """
        Remove duplicate SNPs,
        i.e. SNPs that are completely the same for all samples and therefore do not add information.
        """
        print('Filter duplicate SNPs')
        uniques, index = np.unique(self.X_full, return_index=True, axis=1)
        self.X_full = uniques[:, np.argsort(index)]
        self.snp_ids = self.snp_ids[np.argsort(index)]

    def load_datasplit_indices(self, data_dir: str, n_outerfolds: int, n_innerfolds: int,
                               test_set_size_percentage: int, val_set_size_percentage: int) -> dict:
        """
        Load the datasplit indices saved during file unification.

        Structure:

        .. code-block:: python

            {
                'outerfold_0': {
                    'innerfold_0': {'train': indices_train, 'val': indices_val},
                    'innerfold_1': {'train': indices_train, 'val': indices_val},
                    ...
                    'innerfold_n': {'train': indices_train, 'val': indices_val},
                    'test': test_indices
                    },
                ...
                'outerfold_m': {
                    'innerfold_0': {'train': indices_train, 'val': indices_val},
                    'innerfold_1': {'train': indices_train, 'val': indices_val},
                    ...
                    'innerfold_n': {'train': indices_train, 'val': indices_val},
                    'test': test_indices
                    }
            }

        Caution: The actual structure depends on the datasplit specified by the user,
        e.g. for a train-val-test split only 'outerfold_0' and its subelements 'innerfold_0' and 'test' exist.

        :param data_dir: data directory where the phenotype and genotype matrix are stored
        :param n_outerfolds: number of outerfolds relevant for nested-cv
        :param n_innerfolds: number of folds relevant for nested-cv and cv-test
        :param test_set_size_percentage: size of the test set relevant for cv-test and train-val-test
        :param val_set_size_percentage: size of the validation set relevant for train-val-test

        :return: dictionary with the above-described structure containing all indices for the specified data split
        """
        print('Load datasplit file')
        # construct variables for further process
        if self.datasplit == 'train-val-test':
            n_outerfolds = 1
            n_innerfolds = 1
            datasplit_params = [val_set_size_percentage, test_set_size_percentage]
        elif self.datasplit == 'cv-test':
            n_outerfolds = 1
            n_innerfolds = n_innerfolds
            datasplit_params = [n_innerfolds, test_set_size_percentage]
        elif self.datasplit == 'nested-cv':
            n_outerfolds = n_outerfolds
            n_innerfolds = n_innerfolds
            datasplit_params = [n_outerfolds, n_innerfolds]
        split_param_string = helper_functions.get_subpath_for_datasplit(datasplit=self.datasplit,
                                                                        datasplit_params=datasplit_params)

        datasplit_indices = {}
        with h5py.File(data_dir + '/' + self.index_file_name, "r") as f:
            # load datasplit indices from index file to ensure comparability between different models
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
    def get_index_file_name(genotype_matrix_name: str, phenotype_matrix_name: str, phenotype: str) -> str:
        """
        Get the name of the file containing the indices for maf filters and data splits

        :param genotype_matrix_name: name of the genotype matrix including datatype ending
        :param phenotype_matrix_name: name of the phenotype matrix including datatype ending
        :param phenotype: name of the phenotype to predict

        :return: name of index file
        """
        return genotype_matrix_name.split('.')[0] + '-' + phenotype_matrix_name.split('.')[0] + '-' + phenotype + '.h5'
