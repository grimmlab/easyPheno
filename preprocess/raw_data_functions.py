import argparse
import pandas as pd
import numpy as np
import h5py
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from pandas_plink import read_plink1_bin
from utils import helper_functions
from preprocess import encoding_functions as enc


def prepare_data_files(arguments: argparse.Namespace):
    """
    First check if genotype file is .h5 file
        NO:     Load genotype and create all required .h5 files
        YES:    First check if all required datasets are available, raise Exception if not.
                Then check if index file exists
                    NO: Load genotype and create all required .h5 files
                    YES: Append all required data splits and ma-filters to index file
    :param arguments:
    :return:
    """
    print('Check if all data files have the required format')
    suffix = arguments.genotype_matrix.split('.')[-1]
    if suffix in ('h5', 'hdf5', 'h5py'):
        check_genotype_h5_file(arguments, enc.get_encoding(arguments))
        print('Genotype file available in required format, check index file now.')
        if check_index_file(arguments):
            print('Index file ' + arguments.genotype_matrix.split('.')[0] + '-'\
                    + arguments.phenotype_matrix.split('.')[0] + '-' + arguments.phenotype + '.h5' + ' already exists.'
                    ' Will append required filters and data splits now.')
            append_index_file(arguments)
            print('Done checking data files. All required datasets are available.')
        else:
            print('Index file ' + arguments.genotype_matrix.split('.')[0] + '-'\
                    + arguments.phenotype_matrix.split('.')[0] + '-' + arguments.phenotype + '.h5' + ' does not fulfill'
                    ' requirements. Will load genotype and phenotype matrix and create new index file.')
            save_all_data_files(arguments)
            print('Done checking data files. All required datasets are available.')
    else:
        print('Genotype file not in required format. Will load genotype matrix and save as .h5 file. Will also create '
              'required index file.')
        save_all_data_files(arguments)
        print('Done checking data files. All required datasets are available.')


def check_genotype_h5_file(arguments: argparse.Namespace, encodings: list):
    """
    Function to check .h5 genotype file. Should contain:
    sample_ids: vector with sample names of genotype matrix,
    snp_ids: vector with SNP identifiers of genotype matrix,
    X_{enc}: (samples x SNPs)-genotype matrix in enc encoding, where enc might refer to:
            '012': additive (number of minor alleles)
            'raw': raw (alleles)
    :param arguments: all arguments specified by the user
    :param encodings: list of needed encodings
    :return:
    """
    with h5py.File(arguments.data_dir + '/' + arguments.genotype_matrix, "r") as f:
        keys = list(f.keys())
        if {'sample_ids', 'snp_ids'}.issubset(keys):
            # check if required encoding is available or can be created
            for elem in encodings:  # TODO what if several base encodings are possible?
                if f'X_{elem}' not in f and f'X_{enc.get_base_encoding(elem)}' not in f:
                    raise Exception('Genotype in ' + elem + ' encoding missing. Can not create required encoding. '
                                                            'See documentation for help')
        else:
            raise Exception('sample_ids and/or snp_ids are missing in' + arguments.genotype_matrix)


def check_index_file(arguments: argparse.Namespace):
    """
    Check if index file is available and if the datasets 'y', 'matched_sample_ids', 'X_index', 'y_index' and
    'ma_frequency' exist.
    :param arguments:
    :return:
    """
    index_file = arguments.data_dir + '/' + arguments.genotype_matrix.split('.')[0] + '-'\
                    + arguments.phenotype_matrix.split('.')[0] + '-' + arguments.phenotype + '.h5'
    if os.path.isfile(index_file):
        matched_datasets = ['y', 'matched_sample_ids', 'X_index', 'y_index', 'non_informative_filter', 'ma_frequency']
        with h5py.File(index_file, 'a') as f:
            if 'matched_data' in f and all(z in f['matched_data'] for z in matched_datasets):
                return True
            else:
                return False
    else:
        return False


def save_all_data_files(arguments: argparse.Namespace):
    """
    Function to prepare and save all required data files:
        - genotype matrix in unified format as .h5 file with,
        - phenotype matrix in unified format as .csv file,
        - file containing maf filter and data split indices as .h5.
    :param arguments: all arguments specified by the user
    """
    print('Load genotype file ' + arguments.data_dir + '/' + arguments.genotype_matrix)
    X, X_ids = check_transform_format_genotype_matrix(arguments=arguments)
    print('Have genotype matrix. Load phenotype ' + arguments.phenotype + ' from ' + arguments.data_dir + '/' +
          arguments.phenotype_matrix)
    y = check_and_load_phenotype_matrix(arguments=arguments)
    print('Have phenotype vector. Start matching genotype and phenotype.')
    X, y, sample_ids, X_index, y_index = genotype_phenotype_matching(X, X_ids, y)
    print('Done matching genotype and phenotype. Create index file now.')
    create_index_file(arguments, X, y, sample_ids, X_index, y_index)


def check_transform_format_genotype_matrix(arguments: argparse.Namespace):
    """
    Function to check the format of the specified genotype matrix.
    Unified genotype matrix will be saved in subdirectory data and named NAME_OF_GENOTYPE_MATRIX.h5
    Unified format of the .h5 file of the genotype matrix required for the further processes:
    mandatory:  sample_ids: vector with sample names of genotype matrix,
                SNP_ids: vector with SNP identifiers of genotype matrix,
                X_{enc}: (samples x SNPs)-genotype matrix in enc encoding, where enc might refer to:
                    '012': additive (number of minor alleles)
                    'raw': raw  (alleles)
    optional:   genotype in additional encodings
    Accepts .h5, .hdf5, .h5py, .csv, PLINK binary and PLINK files. .h5, .hdf5, .h5py files must satisfy the unified
    format. If the genotype matrix contains constant SNPs, those will be removed and a new file will be saved.
    Will open .csv, PLINK and binary PLINK files and generate required .h5 format.
    :param arguments: all arguments specified by the user
    :return: sample_ids, X_012
    """
    suffix = arguments.genotype_matrix.split('.')[-1]
    encoding = enc.get_encoding(arguments)
    if suffix in ('h5', 'hdf5', 'h5py'):
        with h5py.File(arguments.data_dir + '/' + arguments.genotype_matrix, "r") as f:
            sample_ids = f['sample_ids'][:]
            if 'X_raw' in f:
                X = f['X_raw'][:]
            elif 'X_012' in f:
                X = f['X_012'][:]
    else:
        if suffix == 'csv':
            sample_ids, snp_ids, X = check_genotype_csv_file(arguments, encoding)

        elif suffix in ('bed', 'bim', 'fam'):
            sample_ids, snp_ids, X = check_genotype_binary_plink_file(arguments)

        elif suffix in ('map', 'ped'):
            sample_ids, snp_ids, X = check_genotype_plink_file(arguments)
        else:
            raise Exception('Only accept .h5, .hdf5, .h5py, .csv, binary PLINK and PLINK genotype files. '
                            'See documentation for help.')
        create_genotype_h5_file(arguments, sample_ids, snp_ids, X)
    return X, sample_ids


def check_genotype_csv_file(arguments: argparse.Namespace, encodings: list):
    """
    Function to load .csv genotype file. File must have the following structure:
    First column must contain the sample ids, the column names should be the SNP ids.
    The values should be the genotype matrix either in additive encoding or in raw encoding.
    If the genotype is in raw encoding, additive encoding will be calculated.
    If genotype is in additive encoding, only this encoding will be returned.
    :param arguments: all arguments specified by the user
    :param encodings: list of needed encodings
    :return: sample ids, SNP ids and genotype in additive and raw encoding (if available)
    """
    gt = pd.read_csv(arguments.data_dir + '/' + arguments.genotype_matrix, index_col=0)
    snp_ids = np.asarray(gt.columns.values)
    sample_ids = np.asarray(gt.index)
    X = np.asarray(gt.values)
    # check encoding of X, only accept additive or raw and check if required encoding can be created
    enc_of_X = enc.check_encoding_of_genotype(X)
    for elem in encodings:
        if elem != enc_of_X and enc.get_base_encoding(elem) != enc_of_X:
            raise Exception('Genotype in ' + arguments.genotype_matrix + ' in wrong encoding. Can not create'
                            ' required encoding. See documentation for help.')
    return sample_ids, snp_ids, X


def check_genotype_binary_plink_file(arguments: argparse.Namespace):
    """
    Function to load binary PLINK file, .bim, .fam, .bed files with same prefix need to be in same folder.
    Compute additive and raw encoding of genotype.
    :param arguments: all arguments specified by the user
    :return: sample ids, SNP ids and genotype in additive and raw encoding
    """
    gt_file = arguments.data_dir + '/' + arguments.genotype_matrix.split(".")[0]
    gt = read_plink1_bin(gt_file + '.bed', gt_file + '.bim', gt_file + '.fam', ref="a0", verbose=False)
    sample_ids = np.array(gt['fid'], dtype=np.int).flatten()
    snp_ids = np.array(gt['snp']).flatten()
    # get raw encoding
    a = np.stack(
        (np.array(gt.a1.values, dtype='S1'), np.zeros(gt.a0.shape, dtype='S1'), np.array(gt.a0.values, dtype='S1')))
    col = np.arange(len(a[0]))
    X_012 = np.array(gt.values)
    X_raw = a[X_012.astype(int), col]
    return sample_ids, snp_ids, X_raw


def check_genotype_plink_file(arguments: argparse.Namespace):
    """
    Function to load PLINK files, .map and .ped file with same prefix need to be in same folder. Accept GENOTYPENAME.ped
    and GENOTYPENAME.map as input.
    Compute additive and raw encoding of genotype.
    :param arguments: all arguments specified by the user
    :return: sample ids, SNP ids and genotype in additive and raw encoding
    """
    gt_file = arguments.data_dir + '/' + arguments.genotype_matrix.split(".")[0]
    with open(gt_file + '.map', 'r') as f:
        SNP_ids = []
        for line in f:
            tmp = line.strip().split(" ")
            SNP_ids.append(tmp[1].strip())
    snp_ids = np.array(SNP_ids)
    iupac_map = {"AA": "A", "GG": "G", "TT": "T", "CC": "C", "AG": "R", "GA": "R", "CT": "Y", "TC": "Y", "GC": "S",
                 "CG": "S", "AT": "W", "TA": "W", "GT": "K", "TG": "K", "AC": "M", "CA": "M"}
    with open(gt_file + '.ped', 'r') as f:
        sample_ids = []
        X_raw = []
        for line in f:
            tmp = line.strip().split(" ")
            sample_ids.append(int(tmp[1].strip()))
            snps = []
            j = 6
            while j < len(tmp) - 1:
                snps.append(iupac_map[tmp[j] + tmp[j + 1]])
                j += 2
            X_raw.append(snps)
    sample_ids = np.array(sample_ids)
    X_raw = np.array(X_raw)
    return sample_ids, snp_ids, X_raw


def create_genotype_h5_file(arguments: argparse.Namespace, sample_ids: np.array, snp_ids: np.array, X: np.array):
    """
    Save genotype matrix in unified .h5 file.
    Structure:
                sample_ids
                snp_ids
                X_raw (or X_012 if X_raw not available)
    :param arguments:
    :param sample_ids: array containing sample ids of genotype data
    :param snp_ids: array containing snp ids of genotype data
    :param X: matrix containing genotype either in raw or in additive encoding
    """
    x_file = arguments.data_dir + '/' + arguments.genotype_matrix.split(".")[0] + '.h5'
    print('Save unified genotype file ' + x_file)
    with h5py.File(x_file, 'w') as f:
        f.create_dataset('sample_ids', data=sample_ids, chunks=True, compression="gzip")
        f.create_dataset('snp_ids', data=snp_ids, chunks=True, compression="gzip")
        encoding = enc.check_encoding_of_genotype(X)
        if encoding == 'raw':
            f.create_dataset('X_raw', data=X, chunks=True, compression="gzip", compression_opts=7)
        elif encoding == '012':
            f.create_dataset('X_012', data=X, chunks=True, compression="gzip", compression_opts=7)
        else:
            raise Exception('Genotype neither in raw or additive encoding. Cannot save .h5 genotype file.')


def check_and_load_phenotype_matrix(arguments: argparse.Namespace):
    """
    Function to check and load the specified phenotype matrix. Only accept .csv, .pheno, .txt files.
    Sample ids need to be in first column, remaining columns should contain phenotypic values
    with phenotype name as column name.
    :param arguments: all arguments specified by the user
    :return: DataFrame with sample_ids as index and phenotype values as single column without NAN values
    """
    suffix = arguments.phenotype_matrix.split('.')[-1]
    if suffix == "csv":
        y = pd.read_csv(arguments.data_dir + '/' + arguments.phenotype_matrix)
    elif suffix in ("pheno", "txt"):
        y = pd.read_csv(arguments.data_dir + '/' + arguments.phenotype_matrix, sep=" ")
    else:
        raise Exception('Only accept .csv, .pheno, .txt phenotype files. See documentation for help')
    y = y.sort_values(y.columns[0]).groupby(y.columns[0]).mean()
    if arguments.phenotype not in y.columns:
        raise Exception('Phenotype ' + arguments.phenotype + ' is not in phenotype file ' + arguments.phenotype_matrix +
                        ' See documentation for help')
    else:
        y = y[[arguments.phenotype]].dropna()
    return y


def genotype_phenotype_matching(X: np.array, X_ids: np.array, y: pd.DataFrame):
    """
    Function to match the handed over genotype and phenotype matrix for the phenotype specified by the user.
    :param X: genotype matrix in additive encoding
    :param X_ids: sample ids of genotype matrix
    :param y: pd.DataFrame containing sample ids of phenotype as index and phenotype values as single column
    :return: matched genotype matrix, matched sample ids, index arrays for genotype and phenotype to redo matching
    """
    y_ids = np.asarray(y.index, dtype=np.int).flatten()
    (y_index, X_index) = (np.reshape(y_ids, (y_ids.shape[0], 1)) == X_ids).nonzero()
    if len(y_index) == 0:
        raise Exception('Samples of genotype and phenotype do not match.')
    X = get_matched_data(X, X_index)
    X_ids = get_matched_data(X_ids, X_index)
    y = get_matched_data(y.values.flatten(), y_index)
    return X, y, X_ids, X_index, y_index


def get_matched_data(data: np.array, index: np.array):
    """
    Function to get elements of data specified in index array
    :param data: matrix or array
    :param index: index array
    :return:
    """
    if data.ndim == 2:
        return data[index, :]
    else:
        return data[index]


def append_index_file(arguments: argparse.Namespace):
    """
    Function to check index file, described in create_index_file(), and append datasets if necessary.
    :param arguments: all arguments specified by the user
    :return:
    """
    with h5py.File(arguments.data_dir + '/' + arguments.genotype_matrix.split('.')[0] + '-'
                   + arguments.phenotype_matrix.split('.')[0] + '-' + arguments.phenotype + '.h5', 'a') as f:
        # check if group 'maf_filter' is available and if user input maf is available, if not: create group/dataset
        if 'maf_filter' not in f:
            maf = f.create_group('maf_filter')
            tmp = (create_maf_filter(arguments.maf_percentage, f['matched_data/ma_frequency'][:]))
            maf.create_dataset(f'maf_{arguments.maf_percentage}', data=tmp, chunks=True, compression="gzip")
        elif f'maf_{arguments.maf_percentage}' not in f['maf_filter']:
            tmp = (create_maf_filter(arguments.maf_percentage, f['matched_data/ma_frequency'][:]))
            f.create_dataset(f'maf_filter/maf_{arguments.maf_percentage}', data=tmp, chunks=True, compression="gzip")
        # check if group datasplit and all user inputs concerning datasplits are available, if not: create all
        if arguments.datasplit == 'nested-cv':
            if 'datasplits' not in f or ('datasplits' in f and 'nested-cv' not in f['datasplits']) or \
                    ('datasplits' in f and 'nested-cv' in f['datasplits'] and
                     f'{helper_functions.get_subpath_for_datasplit(arguments, arguments.datasplit)}' not in
                     f['datasplits/nested-cv']):
                nest = f.create_group(f'datasplits/nested-cv/'
                                      f'{helper_functions.get_subpath_for_datasplit(arguments, arguments.datasplit)}')
                for outer in range(arguments.n_outerfolds):
                    index_dict = check_train_test_splits(f['matched_data/y'], 'nested-cv',
                                                         [arguments.n_outerfolds, arguments.n_innerfolds])
                    o = nest.create_group(f'outerfold_{outer}')
                    o.create_dataset('test', data=index_dict[f'outerfold_{outer}_test'], chunks=True,
                                     compression="gzip")
                    for inner in range(arguments.n_innerfolds):
                        i = o.create_group(f'innerfold_{inner}')
                        i.create_dataset('train', data=index_dict[f'outerfold_{outer}'][f'fold_{inner}_train'],
                                         chunks=True,
                                         compression="gzip")
                        i.create_dataset('val', data=index_dict[f'outerfold_{outer}'][f'fold_{inner}_test'],
                                         chunks=True,
                                         compression="gzip")
        elif arguments.datasplit == 'cv-test':
            if 'datasplits' not in f or ('datasplits' in f and 'cv-test' not in f['datasplits']) or \
                    ('datasplits' in f and 'cv-test' in f['datasplits'] and
                     f'{helper_functions.get_subpath_for_datasplit(arguments, arguments.datasplit)}' not in
                     f['datasplits/cv-test']):
                cv = f.create_group(f'datasplits/cv-test/'
                                    f'{helper_functions.get_subpath_for_datasplit(arguments, arguments.datasplit)}')
                index_dict, test = check_train_test_splits(f['matched_data/y'], 'cv-test',
                                                           [arguments.n_innerfolds, arguments.test_set_size_percentage])
                o = cv.create_group('outerfold_0')
                o.create_dataset('test', data=test, chunks=True, compression="gzip")
                for fold in range(arguments.n_innerfolds):
                    i = o.create_group(f'innerfold_{fold}')
                    i.create_dataset('train', data=index_dict[f'fold_{fold}_train'], chunks=True, compression="gzip")
                    i.create_dataset('val', data=index_dict[f'fold_{fold}_test'], chunks=True, compression="gzip")
        elif arguments.datasplit == 'train-val-test':
            if 'datasplits' not in f or ('datasplits' in f and 'train-val-test' not in f['datasplits']) or \
                    ('datasplits' in f and 'train-val-test' in f['datasplits'] and
                     f'{helper_functions.get_subpath_for_datasplit(arguments, arguments.datasplit)}' not in
                     f['datasplits/cv-test']):
                tvt = f.create_group(f'datasplits/train-val-test/'
                                     f'{helper_functions.get_subpath_for_datasplit(arguments, arguments.datasplit)}')
                train, val, test = check_train_test_splits(f['matched_data/y'], 'train-val-test',
                                        [arguments.validation_set_size_percentage, arguments.test_set_size_percentage])
                o = tvt.create_group('outerfold_0')
                o.create_dataset('test', data=test, chunks=True, compression="gzip")
                i = o.create_group('innerfold_0')
                i.create_dataset('train', data=train, chunks=True, compression="gzip")
                i.create_dataset('val', data=val, chunks=True, compression="gzip")


def create_index_file(arguments: argparse.Namespace, X: np.array, y: np.array,  sample_ids: np.array, X_index: np.array,
                      y_index: np.array):
    """
    Function to create the .h5 file containing the maf filters and data splits for the combination of genotype matrix,
    phenotype matrix and phenotype.
    It will be created using standard values additionally to user inputs for the maf filters and data splits.
    Unified format of .h5 file containing the maf filters and data splits:
        'matched_data': {
                'y': matched phenotypic values,
                'matched_sample_ids': sample ids of matched genotype/phenotype,
                'X_index': indices of genotype matrix to redo matching,
                'y_index': indices of phenotype vector to redo matching,
                'ma_frequency': minor allele frequency of each SNP in genotype file to create new MAF filters
                }
        'maf_filter': {
                'maf_{maf_percentage}': indices of SNPs to delete (with MAF < maf_percentage),
                ...
                }
        'datasplits': {
                'nested_cv': {
                        '#outerfolds-#innerfolds': {
                                'outerfold_0': {
                                    'innerfold_0': {'train': indices_train, 'val': indices_val},
                                    ...
                                    'innerfold_n': {'train': indices_train, 'val': indices_val},
                                    'test': test_indices
                                    },
                                ...
                                'outerfold_m': {
                                    'innerfold_0': {'train': indices_train, 'val': indices_val},
                                    ...
                                    'innerfold_n': {'train': indices_train, 'val': indices_val},
                                    'test': test_indices
                                    }
                                },
                        ...
                        }
                'cv-test': {
                        '#folds-test_percentage': {
                                'outerfold_0': {
                                    'innerfold_0': {'train': indices_train, 'val': indices_val},
                                    ...
                                    'innerfold_n': {'train': indices_train, 'val': indices_val},
                                    'test': test_indices
                                    }
                                },
                        ...
                        }
                'train-val-test': {
                        'train_percentage-val_percentage-test_percentage': {
                                'outerfold_0': {
                                    'innerfold_0': {'train': indices_train, 'val': indices_val},
                                    'test': test_indices
                                    }
                                },
                        ...
                        }
                }

    Standard values for the maf filters and data splits:
        maf thresholds: 1, 3, 5
        folds (inner-/outerfolds for 'nested-cv' and folds for 'cv-test'): 5
        test percentage (for 'cv-test' and 'train-val-test'): 20
        val percentage (for 'train-val-test'): 20
    :param arguments: all arguments specified by the user
    :param X: genotype in additive encoding to create ma-frequencies
    :param y: matched phenotype values
    :param sample_ids: matched sample ids of genotype/phenotype
    :param X_index: index file of genotype to redo matching
    :param y_index: index file of phenotype to redo matching
    :return:
    """
    X, filter = filter_non_informative_snps(X)
    freq = get_minor_allele_freq(X)
    maf_threshold = [0, 1, 3, 5]  # standard values for maf threshold
    if arguments.maf_percentage not in maf_threshold:  # add user input if needed
        maf_threshold.append(arguments.maf_percentage)
    param_nested = [[5, 5]]  # standard values for outer and inner folds for nested-cv
    param_cv = [[5, 20]]  # standard number of folds and test percentage for cv-test
    param_tvt = [[20, 20]]  # standard train and val percentages for train-val-test split
    param_nested = check_datasplit_user_input(arguments, 'nested-cv', param_nested)
    param_cv = check_datasplit_user_input(arguments, 'cv-test', param_cv)
    param_tvt = check_datasplit_user_input(arguments, 'train-val-test', param_tvt)

    with h5py.File(arguments.data_dir + '/' + arguments.genotype_matrix.split('.')[0] + '-'
                   + arguments.phenotype_matrix.split('.')[0] + '-' + arguments.phenotype + '.h5', 'w') as f:
        # all data needed to redo matching of X and y and to create new mafs and new data splits
        data = f.create_group('matched_data')
        data.create_dataset('y', data=y, chunks=True, compression="gzip")
        data.create_dataset('matched_sample_ids', data=sample_ids, chunks=True, compression="gzip")
        data.create_dataset('X_index', data=X_index, chunks=True, compression="gzip")
        data.create_dataset('y_index', data=y_index, chunks=True, compression="gzip")
        data.create_dataset('non_informative_filter', data=filter, chunks=True, compression="gzip")
        data.create_dataset('ma_frequency', data=freq, chunks=True, compression="gzip")
        # create and save standard mafs and maf according to user input
        maf = f.create_group('maf_filter')
        for threshold in maf_threshold:
            tmp = (create_maf_filter(threshold, freq))
            maf.create_dataset(f'maf_{threshold}', data=tmp, chunks=True, compression="gzip")
        # create and save standard data splits and splits according to user input
        dsplit = f.create_group('datasplits')
        nest = dsplit.create_group('nested-cv')
        for elem in param_nested:
            n = nest.create_group(helper_functions.get_subpath_for_datasplit(arguments, 'nested-cv',
                                                                             additional_param=elem))
            for outer in range(elem[0]):
                index_dict = check_train_test_splits(y, 'nested-cv', elem)
                o = n.create_group(f'outerfold_{outer}')
                o.create_dataset('test', data=index_dict[f'outerfold_{outer}_test'], chunks=True, compression="gzip")
                for inner in range(elem[1]):
                    i = o.create_group(f'innerfold_{inner}')
                    i.create_dataset('train', data=index_dict[f'outerfold_{outer}'][f'fold_{inner}_train'], chunks=True,
                                     compression="gzip")
                    i.create_dataset('val', data=index_dict[f'outerfold_{outer}'][f'fold_{inner}_test'], chunks=True,
                                     compression="gzip")
        cv = dsplit.create_group('cv-test')
        for elem in param_cv:
            index_dict, test = check_train_test_splits(y, 'cv-test', elem)
            n = cv.create_group(helper_functions.get_subpath_for_datasplit(arguments, 'cv-test', additional_param=elem))
            o = n.create_group('outerfold_0')
            o.create_dataset('test', data=test, chunks=True, compression="gzip")
            for fold in range(elem[0]):
                i = o.create_group(f'innerfold_{fold}')
                i.create_dataset('train', data=index_dict[f'fold_{fold}_train'], chunks=True, compression="gzip")
                i.create_dataset('val', data=index_dict[f'fold_{fold}_test'], chunks=True, compression="gzip")
        tvt = dsplit.create_group('train-val-test')
        for elem in param_tvt:
            train, val, test = check_train_test_splits(y, 'train-val-test', elem)
            n = tvt.create_group(helper_functions.get_subpath_for_datasplit(arguments, 'train-val-test',
                                                                            additional_param=elem))
            o = n.create_group('outerfold_0')
            o.create_dataset('test', data=test, chunks=True, compression="gzip")
            i = o.create_group('innerfold_0')
            i.create_dataset('train', data=train, chunks=True, compression="gzip")
            i.create_dataset('val', data=val, chunks=True, compression="gzip")


def filter_non_informative_snps(X: np.array):
    """
    Function to remove non-informative SNPs, i.e. SNPs that are constant.
    :param X: genotype matrix in raw or additive encoding
    :return: filtered genotype matrix and filter-vector
    """
    tmp = X == X[0, :]
    X_filtered = X[:, ~tmp.all(0)]
    return X_filtered, (~tmp.all(0)).nonzero()[0]


def get_minor_allele_freq(X: np.array):
    """
    Function to compute minor allele frequencies of genotype matrix
    :param X: genotype matrix in additive encoding
    :return: array with frequencies
    """
    encoding = enc.check_encoding_of_genotype(X)
    if encoding == '012':
        return (np.sum(X, 0)) / (2 * X.shape[0])
    else:
        freq = []
        for i, col in enumerate(np.transpose(X)):
            unique, counts = np.unique(col, return_counts=True)
            if len(unique) > 3:
                raise Exception('More than two alleles encountered at SNP ', i)
            elif len(unique) == 3:
                boolean = (unique.astype(str) == 'A') | (unique.astype(str) == 'T') | (unique.astype(str) == 'C') | (
                            unique.astype(str) == 'G')
                homozygous = unique[boolean]
                hetero = unique[~boolean][0]
                pairs = [['A', 'C'], ['A', 'G'], ['A', 'T'], ['C', 'G'], ['C', 'T'], ['G', 'T']]
                heterozygous_nuc = ['M', 'R', 'W', 'S', 'Y', 'K']
                for j, pair in enumerate(pairs):
                    if all(h in pair for h in homozygous) and hetero != heterozygous_nuc[j]:
                        raise Exception('More than two alleles encountered at SNP ' + str(i))
                freq.append((np.min(counts[boolean]) + 0.5 * counts[~boolean][0]) / len(col))
            else:
                freq.append(np.min(counts) / len(col))
        return np.array(freq)


def create_maf_filter(maf: int, freq: np.array):
    """
    Function to compute minor allele frequency filter
    :param maf: maf threshold as percentage value
    :param freq: array containing minor allele frequencies as decimal value
    :return: array containing indices of SNPs with MAF smaller than specified threshold, i.e. SNPs to delete
    """
    return np.where(freq <= (maf / 100))[0]


def check_datasplit_user_input(arguments: argparse.Namespace, split: str, param: list):
    """
    Function to check if user input of data split parameters differs from standard values. If it does, adds
    input to list of parameters.
    :param arguments: all arguments specified by the user
    :param split: type of data split
    :param param: standard parameters to compare to
    :return: adapted list of parameters
    """
    if split == 'nested-cv':
        user_input = [arguments.n_outerfolds, arguments.n_innerfolds]
    elif split == 'cv-test':
        user_input = [arguments.n_innerfolds, arguments.test_set_size_percentage]
    elif split == 'train-val-test':
        user_input = [arguments.validation_set_size_percentage, arguments.test_set_size_percentage]
    else:
        raise Exception('Only accept nested-cv, cv-test or train-val-test as data splits.')
    if arguments.datasplit == split and user_input not in param:
        param.append(user_input)
    return param


def check_train_test_splits(y: np.array, split: str, param: list):
    """
    Function to create stratified train-test splits. Continuous values will be grouped into bins and stratified
    according to those.
    :param split: type of data split ('nested-cv', 'cv-test', 'train-val-test')
    :param y: array with phenotypic values for stratification
    :param param: parameters to use for split:
    [n_outerfolds, n_innerfolds] for nested-cv
    [n_innerfolds, test_set_size_percentage] for cv-test
    [validation_set_size_percentage, test_set_size_percentage] for train-val-test
    :return: index arrays for splits
    """
    y_binned = make_bins(y, split, param)
    if split == 'nested-cv':
        return make_nested_cv(y=y_binned, outerfolds=param[0], innerfolds=param[1])
    elif split == 'cv-test':
        x_train, x_test, y_train = make_train_test_split(y=y_binned, test_size=param[1], val=False)
        cv_dict = make_stratified_cv(x=x_train, y=y_train, split_number=param[0])
        return cv_dict, x_test
    elif split == 'train-val-test':
        return make_train_test_split(y=y_binned, test_size=param[1], val_size=param[0], val=True)
    else:
        raise Exception('Only accept nested-cv, cv-test or train-val-test as data splits.')


def make_bins(y: np.array, split: str, param: list):
    """
    Function to create bins of continuous values for stratification.
    :param y: array containing phenotypic values
    :param split: train test split to use
    :param param: list of parameters to use:
    [n_outerfolds, n_innerfolds] for nested-cv
    [n_innerfolds, test_set_size_percentage] for cv-test
    :return: binned array
    """
    if helper_functions.test_likely_categorical(y):
        return y.astype(int)
    else:
        if split == 'nested-cv':
            tmp = len(y)/(param[0] + param[1])
        elif split == 'cv-test':
            tmp = len(y)*(1-param[1]/100)/param[0]
        else:
            tmp = len(y)/10 + 1

        number_of_bins = min(int(tmp) - 1, 10)
        edges = np.percentile(y, np.linspace(0, 100, number_of_bins)[1:])
        y_binned = np.digitize(y, edges, right=True)
        return y_binned


def make_nested_cv(y: np.array, outerfolds: int, innerfolds: int):
    """
    Function that creates index dictionary for stratified nested cross validation with the following structure:
        {'outerfold_0_test': test_indices,
        'outerfold_0': {fold_0_train: innerfold_0_train_indices,
                        fold_0_test: innerfold_0_test_indices,
                        ...
                        fold_n_train: innerfold_n_train_indices,
                        fold_n_test: innerfold_n_test_indices
                        },
        ...
        'outerfold_m_test': test_indices,
        'outerfold_m': {fold_0_train: innerfold_0_train_indices,
                        fold_0_test: innerfold_0_test_indices,
                        ...
                        fold_n_train: innerfold_n_train_indices,
                        fold_n_test: innerfold_n_test_indices
                        }
        }
    :param y: target values grouped in bins for stratification
    :param outerfolds: number of outer folds
    :param innerfolds: number of inner folds
    :return: index dictionary
    """
    outer_cv = StratifiedKFold(n_splits=outerfolds)
    index_dict = {}
    outer_fold = 0
    for train_index, test_index in outer_cv.split(np.zeros(len(y)), y):
        np.random.shuffle(test_index)
        index_dict[f'outerfold_{outer_fold}_test'] = test_index
        index_dict[f'outerfold_{outer_fold}'] = make_stratified_cv(train_index, y[train_index], split_number=innerfolds)
        outer_fold += 1
    return index_dict


def make_stratified_cv(x: np.array, y: np.array, split_number: int):
    """
    Function to create index dictionary for stratified cross-validation with following structure:
    {fold_0_train: fold_0_train_indices,
    fold_0_test: fold_0_test_indices,
    ...
    fold_n_train: fold_n_train_indices,
    fold_n_test: fold_n_test_indices
    }
    :param x:
    :param y: target values binned in groups for stratification
    :param split_number: number of folds
    :return: dictionary containing train and validation indices for each fold
    """
    cv = StratifiedKFold(n_splits=split_number)
    index_dict = {}
    fold = 0
    for train_index, test_index in cv.split(x, y):
        np.random.shuffle(train_index)
        np.random.shuffle(test_index)
        index_dict[f'fold_{fold}_train'] = train_index
        index_dict[f'fold_{fold}_test'] = test_index
        fold += 1
    return index_dict


def make_train_test_split(y: np.array, test_size: int, val_size=None, val=False, random=42):
    """
    Function to create index arrays for stratified train-test, respectively train-val-test splits.
    :param y: target values grouped in bins for stratification
    :param test_size: size of test set as percentage value
    :param val_size: size of validation set as percentage value
    :param val: default=False, if True, function returns validation set additionally to train and test set.
    :param random: random number, default=42, controls shuffling of data
    :return: either train, val and test index arrays or
    train and test index arrays and corresponding binned target values
    """
    x = np.arange(len(y))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size/100, stratify=y, random_state=random)
    if not val:
        return x_train, x_test, y_train
    else:
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size/100, stratify=y_train,
                                                          random_state=random)
        return x_train, x_val, x_test
