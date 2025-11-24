import h5py
import numpy as np
import sklearn.preprocessing
import pathlib

from ..utils import helper_functions
from . import raw_data_functions, encoding_functions


class Basedata:
    """
    Parent class for all dataset classes.

    **Attributes**

        - inference_only (*bool*): set to True if only inference is requested
        - encoding (*str*): the encoding to use (standard encoding or user-defined)
        - X_full (*numpy.array*): all (matched, maf- and duplicated-filtered) SNPs
        - y_full (*numpy.array*): all target values
        - sample_ids_full (*numpy.array*):all sample ids
        - snp_ids (*numpy.array*): SNP ids
        - datasplit (*str*): datasplit to use
        - datasplit_indices (*dict*): dictionary containing all indices for the specified datasplit

    :param data_dir: data directory where the phenotype and genotype matrix are stored
    :param genotype_matrix_name: name of the genotype matrix including datatype ending
    :param encoding: the encoding to use (standard encoding or user-defined)
    :param maf_percentage: threshold for MAF filter as percentage value
    :param do_snp_filters: specify if SNP filters (e.g. duplicates, maf etc.) should be applied
    """

    def __init__(self, data_dir: pathlib.Path, genotype_matrix_name: str, encoding: str, maf_percentage: int,
                 do_snp_filters: bool = True):
        self.inference_only = False
        self.encoding = encoding
        self.index_file_name = None
        self.X_full = None
        self.y_full = None
        self.sample_ids_full = None
        self.snp_ids = None
        self.datasplit = None
        self.datasplit_indices = None

    def load_match_raw_data(self, data_dir: pathlib.Path, genotype_matrix_name: str, do_snp_filters: bool = True):
        """
        Load the full genotype and phenotype matrices specified and match them

        :param data_dir: data directory where the phenotype and genotype matrix are stored
        :param genotype_matrix_name: name of the genotype matrix including datatype ending
        :param do_snp_filters: specify if SNP filters (e.g. duplicates, maf etc.) should be applied

        :return: matched genotype, phenotype and sample ids
        """
        print('Load and match raw data')
        with h5py.File(data_dir.joinpath(self.index_file_name), "r") as f:
            # Load information from index file
            if not self.inference_only:
                y = f['matched_data/y'][:]  # TODO change if multiple phenotypes
                if helper_functions.test_likely_categorical(y):
                    if y.dtype.type is np.float64:
                        y = y.astype(int)
                    y = sklearn.preprocessing.LabelEncoder().fit_transform(y)
            sample_ids = f['matched_data/matched_sample_ids'][:].astype(str)
            non_informative_filter = f['matched_data/non_informative_filter'][:]
            X_index = f['matched_data/X_index'][:]

        with h5py.File(data_dir.joinpath(genotype_matrix_name).with_suffix('.h5'), "r") as f:
            # Load genotype data
            snp_ids = f['snp_ids'][non_informative_filter].astype(str) if do_snp_filters \
                else f['snp_ids'][:].astype(str)
            if f'X_{self.encoding}' in f:
                X = f[f'X_{self.encoding}'][:, non_informative_filter] if do_snp_filters else f[f'X_{self.encoding}'][:]
            elif f'X_{encoding_functions.get_base_encoding(encoding=self.encoding)}' in f:
                X_base = \
                    f[f'X_{encoding_functions.get_base_encoding(encoding=self.encoding)}'][:, non_informative_filter] \
                        if do_snp_filters else f[f'X_{encoding_functions.get_base_encoding(encoding=self.encoding)}'][:]
                X = encoding_functions.encode_genotype(X=X_base, required_encoding=self.encoding)
            else:
                raise Exception('Genotype in ' + self.encoding + ' encoding missing. Can not create required encoding. '
                                                                 'See documentation for help')
        X = raw_data_functions.get_matched_data(data=X, index=X_index)
        # sanity checks
        raw_data_functions.check_genotype_shape(X=X, sample_ids=sample_ids, snp_ids=snp_ids)
        if raw_data_functions.check_duplicate_samples(sample_ids=sample_ids):
            raise Exception('The genotype matrix contains duplicate samples. Please check again.')
        if not self.inference_only:
            return X, np.reshape(y, (-1, 1)), np.reshape(sample_ids, (-1, 1)), snp_ids
        else:
            return X, np.reshape(sample_ids, (-1, 1)), snp_ids

    def maf_filter_raw_data(self, data_dir: pathlib.Path, maf_percentage: int):
        """
        Apply maf filter to full raw data, if maf=0 only non-informative SNPs will be removed

        :param data_dir: data directory where the phenotype and genotype matrix are stored
        :param maf_percentage: threshold for MAF filter as percentage value
        """
        print('Apply MAF filter')
        with h5py.File(data_dir.joinpath(self.index_file_name), "r") as f:
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
        self.X_full = self.X_full[:, np.sort(index)]
        self.snp_ids = self.snp_ids[np.sort(index)]

    def check_and_save_filtered_snp_ids(self, data_dir: pathlib.Path, maf_percentage: int):
        """
        Check if snp_ids for specific maf percentage and encoding are saved in index_file.
        If not, save them in 'matched_data/final_snp_ids/{encoding}/maf_{maf_percentage}_snp_ids'

        :param data_dir: data directory where the phenotype and genotype matrix are stored
        :param maf_percentage: threshold for MAF filter as percentage value
        """
        print('Check if final snp_ids already exist in index_file for used encoding and maf percentage. '
              'Save them if necessary.')
        with h5py.File(data_dir.joinpath(self.index_file_name), "a") as f:
            if 'final_snp_ids' not in f['matched_data']:
                final = f.create_group('matched_data/final_snp_ids')
                enc = final.create_group(f'{self.encoding}')
                enc.create_dataset(f'maf_{maf_percentage}_snp_ids',
                                   data=self.snp_ids.astype(bytes), chunks=True, compression="gzip")
            elif f'{self.encoding}' not in f['matched_data/final_snp_ids']:
                enc = f.create_group(f'matched_data/final_snp_ids/{self.encoding}')
                enc.create_dataset(f'maf_{maf_percentage}_snp_ids',
                                   data=self.snp_ids.astype(bytes), chunks=True, compression="gzip")
            elif f'maf_{maf_percentage}_snp_ids' not in f[f'matched_data/final_snp_ids/{self.encoding}']:
                f.create_dataset(f'matched_data/final_snp_ids/{self.encoding}/maf_{maf_percentage}_snp_ids',
                                 data=self.snp_ids.astype(bytes), chunks=True, compression="gzip")

    @staticmethod
    def get_index_file_name(genotype_matrix_name: str, phenotype_matrix_name: str, phenotype: str) -> str:
        """
        Get the name of the file containing the indices for maf filters and data splits

        :param genotype_matrix_name: name of the genotype matrix including datatype ending
        :param phenotype_matrix_name: name of the phenotype matrix including datatype ending
        :param phenotype: name of the phenotype to predict

        :return: name of index file
        """
        if phenotype_matrix_name is not None:
            return genotype_matrix_name.split('.')[0] + '-' + phenotype_matrix_name.split('.')[0] + '-' + phenotype + '.h5'
        else:
            return genotype_matrix_name.split('.')[0] + '-inference_only.h5'


class Dataset(Basedata):
    """
    Class containing dataset ready for optimization (e.g. geno/phenotype matched). Based on parent class Basedata

    **Attributes**

        - inference_only (*bool*): False
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
    :param do_snp_filters: specify if SNP filters (e.g. duplicates, maf etc.) should be applied
    """

    def __init__(self, data_dir: pathlib.Path, genotype_matrix_name: str, phenotype_matrix_name: str, phenotype: str,
                 datasplit: str, n_outerfolds: int, n_innerfolds: int, test_set_size_percentage: int,
                 val_set_size_percentage: int, encoding: str, maf_percentage: int, do_snp_filters: bool = True):

        super().__init__(data_dir=data_dir, genotype_matrix_name=genotype_matrix_name, encoding=encoding,
                         maf_percentage=maf_percentage, do_snp_filters=do_snp_filters)
        self.index_file_name = self.get_index_file_name(genotype_matrix_name=genotype_matrix_name,
                                                        phenotype_matrix_name=phenotype_matrix_name,
                                                        phenotype=phenotype)
        self.datasplit = datasplit
        self.X_full, self.y_full, self.sample_ids_full, self.snp_ids = self.load_match_raw_data(
            data_dir=data_dir, genotype_matrix_name=genotype_matrix_name, do_snp_filters=do_snp_filters)
        if do_snp_filters:
            self.maf_filter_raw_data(data_dir=data_dir, maf_percentage=maf_percentage)
            self.filter_duplicate_snps()
        self.check_and_save_filtered_snp_ids(data_dir=data_dir, maf_percentage=maf_percentage)
        self.datasplit_indices = self.load_datasplit_indices(
            data_dir=data_dir, n_outerfolds=n_outerfolds, n_innerfolds=n_innerfolds,
            test_set_size_percentage=test_set_size_percentage, val_set_size_percentage=val_set_size_percentage
        )
        self.check_datasplit(
            n_outerfolds=1 if datasplit != 'nested-cv' else n_outerfolds,
            n_innerfolds=1 if datasplit == 'train-val-test' else n_innerfolds
        )


    def load_datasplit_indices(self, data_dir: pathlib.Path, n_outerfolds: int, n_innerfolds: int,
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
        with h5py.File(data_dir.joinpath(self.index_file_name), "r") as f:
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

    def check_datasplit(self, n_outerfolds: int, n_innerfolds: int):
        """
        Check if the datasplit is valid. Raise Exceptions if train, val or test sets contain same samples.

        :param n_outerfolds: number of outerfolds in datasplit_indices dictionary
        :param n_innerfolds: number of folds in datasplit_indices dictionary
        """
        all_sample_ids_test = []
        for j in range(n_outerfolds):
            sample_ids_test = set(self.sample_ids_full[self.datasplit_indices[f'outerfold_{j}']['test']].flatten())
            all_sample_ids_test.extend(sample_ids_test)
            for i in range(n_innerfolds):
                sample_ids_train = set(
                    self.sample_ids_full[self.datasplit_indices[f'outerfold_{j}'][f'innerfold_{i}']['train']].flatten()
                )
                sample_ids_val = set(
                    self.sample_ids_full[self.datasplit_indices[f'outerfold_{j}'][f'innerfold_{i}']['val']].flatten())
                if len(sample_ids_train.intersection(sample_ids_val)) != 0:
                    raise Exception(
                        'Something with the datasplit went wrong - the intersection of train and val samples is not '
                        'empty. Please check again.'
                    )
                if len(sample_ids_train.intersection(sample_ids_test)) != 0:
                    raise Exception(
                        'Something with the datasplit went wrong - the intersection of train and test samples is not '
                        'empty. Please check again.'
                    )
                if len(sample_ids_val.intersection(sample_ids_test)) != 0:
                    raise Exception(
                        'Something with the datasplit went wrong - the intersection of val and test samples is not '
                        'empty. Please check again.'
                    )
        if self.datasplit == 'nested-cv':
            if len(set(all_sample_ids_test).intersection(set(self.sample_ids_full.flatten()))) \
                    != len(set(self.sample_ids_full.flatten())):
                raise Exception('Something with the datasplit went wrong - '
                                'not all sample ids are in one of the outerfold test sets')
        print('Checked datasplit for all folds.')


class Datasetinfonly(Basedata):
    """
    Class containing dataset for inference only. Based on parent class Basedata

    **Attributes**

        - inference_only (*bool*): True
        - encoding (*str*): the encoding to use (standard encoding or user-defined)
        - X_full (*numpy.array*): all (matched, maf- and duplicated-filtered) SNPs
        - sample_ids_full (*numpy.array*):all sample ids
        - snp_ids (*numpy.array*): SNP ids
        - y_full (*numpy.array*): None
        - datasplit (*str*): None
        - datasplit_indices (*dict*): None

    :param data_dir: data directory where the phenotype and genotype matrix are stored
    :param genotype_matrix_name: name of the genotype matrix including datatype ending
    :param maf_percentage: threshold for MAF filter as percentage value
    :param encoding: the encoding to use (standard encoding or user-defined)
    :param do_snp_filters: specify if SNP filters (e.g. duplicates, maf etc.) should be applied
    """

    def __init__(self, data_dir: pathlib.Path, genotype_matrix_name: str, encoding: str, maf_percentage: int,
                 do_snp_filters: bool = True):

        super().__init__(data_dir=data_dir, genotype_matrix_name=genotype_matrix_name, encoding=encoding,
                         maf_percentage=maf_percentage, do_snp_filters=do_snp_filters)
        self.inference_only = True
        self.index_file_name = self.get_index_file_name(genotype_matrix_name=genotype_matrix_name,
                                                        phenotype_matrix_name=None, phenotype=None)
        self.X_full, self.sample_ids_full, self.snp_ids = self.load_match_raw_data(
            data_dir=data_dir, genotype_matrix_name=genotype_matrix_name, do_snp_filters=do_snp_filters)
        if do_snp_filters:
            self.maf_filter_raw_data(data_dir=data_dir, maf_percentage=maf_percentage)
            self.filter_duplicate_snps()
        self.check_and_save_filtered_snp_ids(data_dir=data_dir, maf_percentage=maf_percentage)
        