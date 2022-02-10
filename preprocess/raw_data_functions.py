import argparse
import pandas as pd
import numpy as np
import h5py
import os
from pandas_plink import read_plink1_bin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from utils.helper_functions import test_likely_categorical
from preprocess.encoding_functions import encode_raw_genotype


def prepare_data_files(arguments: argparse.Namespace):
    """
    Function to prepare and save all required data files:
        - genotype matrix in unified format as .h5 file with,
        - phenotype matrix in unified format as .csv file,
        - file containing maf filter and data split indices as .h5.
    :param arguments: all arguments specified by the user
    """
    X, X_ids = check_transform_format_genotype_matrix(arguments=arguments)
    y = check_and_load_phenotype_matrix(arguments=arguments)
    X, y, sample_ids, X_index, y_index = genotype_phenotype_matching(X, X_ids, y)
    check_create_index_file(arguments, X, y, sample_ids, X_index, y_index)


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
    encoding = []  # TODO Liste aller Codierungen die überprüft werden müssen
    if suffix in ('h5', 'hdf5', 'h5py'):
        sample_ids, snp_ids, X_012, X_raw = check_genotype_h5_file(arguments, encoding)
    else:
        if suffix == 'csv':
            sample_ids, snp_ids, X_012, X_raw = check_genotype_csv_file(arguments, encoding)

        elif suffix in ('bed', 'bim', 'fam'):
            sample_ids, snp_ids, X_012, X_raw = check_genotype_binary_plink_file(arguments)

        elif suffix in ('map', 'ped'):
            sample_ids, snp_ids, X_012, X_raw = check_genotype_plink_file(arguments)
        else:
            raise Exception('Only accept .h5, .hdf5, .h5py, .csv, binary PLINK and PLINK genotype files. '
                            'See documentation for help.')
        X_012, X_raw, snp_ids = filter_non_informative_snps(X_012, X_raw, snp_ids)
    if snp_ids is not None:
        create_genotype_h5_file(arguments, sample_ids, snp_ids, X_012, X_raw)
    return X_012, sample_ids


def check_genotype_h5_file(arguments: argparse.Namespace, encoding: list):
    """
    Function to load and check .h5 genotype file. Should contain:
    sample_ids: vector with sample names of genotype matrix,
    snp_ids: vector with SNP identifiers of genotype matrix,
    X_{enc}: (samples x SNPs)-genotype matrix in enc encoding, where enc might refer to:
            '012': additive (number of minor alleles)
            'raw': raw (alleles)
    In order to work with other encodings X_raw is required.
    If genotype matrix contains non-informative SNPs (with standard deviation = 0) returns matrix without those SNPs
    :param arguments: all arguments specified by the user
    :param encoding: list of needed encodings
    :return: sample_ids, snp_ids, X_012, X_raw;
    snp_ids and X_raw might be None if matrix does not contain non-informative SNPs
    """
    with h5py.File(arguments.base_dir + '/data/' + arguments.genotype_matrix, "r") as f:
        keys = list(f.keys())
        if {'sample_ids', 'snp_ids'}.issubset(keys):
            sample_ids = f['sample_ids'][:]
            snp_ids = f['snp_ids'][:]
            if 'X_raw' in keys:
                    X_raw = f['X_raw'][:]
                    X_012 = encode_raw_genotype(X_raw, '012')
            elif any(z in encoding for z in ['raw', 'onehot']):  # TODO adapt when we have additional encodings
                raise Exception('Genotype in raw encoding missing in ' + arguments.genotype_matrix +
                                '. Can not create required encodings. See documentation for help.')
            elif 'X_012' in keys:
                X_012 = f['X_012'][:]
                X_raw = None
            else:
                raise Exception('No genotype matrix in file ' + arguments.genotype_matrix + '. Need genotype matrix in '
                                'raw or additive encoding. See documentation for help')
            m = X_012.shape[1]
            X_012, X_raw, snp_ids = filter_non_informative_snps(X_012, X_raw, snp_ids)
            if X_012.shape[1] != m:
                return sample_ids, snp_ids, X_012, X_raw
            else:
                return sample_ids, None, X_012, None
        else:
            raise Exception('keys "sample_ids" and/or "SNP_ids" are missing in' + arguments.genotype_matrix)


def check_genotype_csv_file(arguments: argparse.Namespace, encoding: list):
    """
    Function to load .csv genotype file. File must have the following structure:
    First column must contain the sample ids, the column names should be the SNP ids.
    The values should be the genotype matrix either in additive encoding or in raw encoding.
    If the genotype is in raw encoding, additive encoding will be calculated.
    If genotype is in additive encoding, only this encoding will be returned.
    :param arguments: all arguments specified by the user
    :param encoding: list of needed encodings
    :return: sample ids, SNP ids and genotype in additive and raw encoding (if available)
    """
    gt = pd.read_csv(arguments.base_dir + '/data/' + arguments.genotype_matrix, index_col=0)
    snp_ids = np.asarray(gt.columns.values)
    sample_ids = np.asarray(gt.index)
    X = np.asarray(gt.values)
    # check encoding of X, only accept additive or raw (alleles)
    unique = np.unique(X)
    if all(z in ['A', 'C', 'G', 'T'] for z in unique):  # TODO heterozygous!?
        X_012 = encode_raw_genotype(X, '012')
        return sample_ids, snp_ids, X_012, X
    elif all(z in [0, 1, 2] for z in unique):
        if any(z in encoding for z in ['raw', 'onehot']):  # TODO adapt for additional encodings
            raise Exception('Genotype in ' + arguments.genotype_matrix + ' not in raw encoding. Can not create'
                            ' required encodings. See documentation for help.')
        return sample_ids, snp_ids, X, None
    else:
        raise Exception('Genotype in ' + arguments.genotype_matrix + ' is neither in additive nor in raw '
                                                                     'encoding. See documentation for help.')


def check_genotype_binary_plink_file(arguments: argparse.Namespace):
    """
    Function to load binary PLINK file, .bim, .fam, .bed files with same prefix need to be in same folder.
    Compute additive and raw encoding of genotype.
    :param arguments: all arguments specified by the user
    :return: sample ids, SNP ids and genotype in additive and raw encoding
    """
    gt_file = arguments.base_dir + '/data/' + arguments.genotype_matrix.split(".")[0]
    gt = read_plink1_bin(gt_file + '.bed', gt_file + '.bim', gt_file + '.fam', ref="a0", verbose=False)
    sample_ids = np.array(gt['fid'], dtype=np.int).flatten()
    snp_ids = np.array(gt['snp']).flatten()
    # get raw encoding
    a = np.stack(
        (np.array(gt.a1.values, dtype='S1'), np.zeros(gt.a0.shape, dtype='S1'), np.array(gt.a0.values, dtype='S1')))
    col = np.arange(len(a[0]))
    X_012 = np.array(gt.values)
    X_raw = a[X_012.astype(int), col]
    return sample_ids, snp_ids, X_012, X_raw


def check_genotype_plink_file(arguments: argparse.Namespace):
    """
    Function to load PLINK files, .map and .ped file with same prefix need to be in same folder. Accept GENOTYPENAME.ped
    and GENOTYPENAME.map as input.
    Compute additive and raw encoding of genotype.
    :param arguments: all arguments specified by the user
    :return: sample ids, SNP ids and genotype in additive and raw encoding
    """
    gt_file = arguments.base_dir + '/data/' + arguments.genotype_matrix.split(".")[0]
    with open(gt_file + '.map', 'r') as f:
        snp_ids = []
        for line in f:
            tmp = line.strip().split(" ")
            snp_ids.append(tmp[1].strip())
    snp_ids = np.array(snp_ids)
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
    X_012 = encode_raw_genotype(X_raw, '012')
    return sample_ids, snp_ids, X_012, X_raw


def filter_non_informative_snps(X_012: np.array, X_raw: np.array, snp_ids: np.array):
    """
    Function to remove constant SNPs, i.e. SNPs where all values are equal.
    :param X_012: genotype matrix in additive encoding
    :param X_raw: genotype matrix in raw encoding
    :param snp_ids: array containing the SNP ids
    :return: filtered genotype matrices and SNP_ids
    """
    tmp = np.where(X_012.std(axis=0) == 0)[0]
    X_012 = np.delete(X_012, tmp, axis=1)
    snp_ids = np.delete(snp_ids, tmp, axis=0)
    if X_raw is not None:
        X_raw = np.delete(X_raw, tmp, axis=1)
    return X_012, X_raw, snp_ids


def create_genotype_h5_file(arguments: argparse.Namespace, sample_ids: np.array, snp_ids: np.array, X_012: np.array,
                            X_raw: np.array):
    """
    Save genotype matrix in unified .h5 file.
    Structure:
                sample_ids
                snp_ids
                X_raw (or X_012 if X_raw not available)
    :param arguments:
    :param sample_ids: array containing sample ids of genotype data
    :param snp_ids: array containing snp ids of genotype data
    :param X_012: matrix containing genotype in additive encoding
    :param X_raw: matrix containing genotype in raw encoding
    """
    with h5py.File(arguments.base_dir + '/data/' + arguments.genotype_matrix.split(".")[0] + '.h5', 'w') as f:
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('sample_ids', data=sample_ids, chunks=True, compression="gzip")
        f.create_dataset('snp_ids', data=snp_ids, chunks=True, compression="gzip", dtype=dt)
        if X_raw is not None:
            f.create_dataset('X_raw', data=X_raw, chunks=True, compression="gzip", compression_opts=7)
        else:
            f.create_dataset('X_012', data=X_012, chunks=True, compression="gzip", compression_opts=7)
    # TODO change name --> even if geno matrix is .h5 file, might need to save new file with filtered snps

def check_and_load_phenotype_matrix(arguments: argparse.Namespace):
    """
    Function to check and load the specified phenotype matrix. Only accept .csv, .pheno, .txt files.
    Sample ids need to be in first column, remaining columns should contain phenotypic values
    with phenotype name as column name. .pheno and .txt files are assumed to have a single space as separator.
    :param arguments: all arguments specified by the user
    :return: DataFrame with sample_ids as index and phenotype values as single column without NAN values
    """
    suffix = arguments.phenotype_matrix.split('.')[-1]
    if suffix == 'csv':
        y = pd.read_csv(arguments.base_dir + '/data/' + arguments.phenotype_matrix)
        y = y.sort_values(y.columns[0]).groupby(y.columns[0]).mean()
    elif suffix in ('pheno', 'txt'):
        y = pd.read_csv(arguments.base_dir + '/data/' + arguments.phenotype_matrix, sep = " ")
        y = y.sort_values(y.columns[0]).groupby(y.columns[0]).mean()
    else:
        raise Exception('Only accept .csv, .pheno, .txt phenotype files. See documentation for help')
    if arguments.phenotype not in y.columns:
        raise Exception('Phenotype ' + arguments.phenotype + ' is not in phenotype file '
                        + arguments.phenotype_matrix + ' See documentation for help')
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


def check_create_index_file(arguments: argparse.Namespace, X: np.array, y: np.array, sample_ids: np.array,
                            X_index: np.array, y_index: np.array):
    """
    Function to check the .h5 file containing the maf filters and data splits for the combination of genotype matrix,
    phenotype matrix and phenotype.
    It will be created using standard values for the maf filters and data splits in case it does not exist.
    Otherwise, the maf filter and data splits specified by the user are checked for existence.
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
    if os.path.isfile(arguments.base_dir + '/data/' + arguments.genotype_matrix.split('.')[0] + '-'
                      + arguments.phenotype_matrix.split('.')[0] + '-' + arguments.phenotype + '.h5'):
        append_index_file(arguments)
    else:
        create_index_file(arguments, X, y, sample_ids, X_index, y_index)


def append_index_file(arguments: argparse.Namespace):
    """
    Function to check index file and append datasets if necessary.
    :param arguments:
    :return:
    """
    matched_datasets = ['y', 'matched_sample_ids', 'X_index', 'y_index', 'ma_frequency']
    with h5py.File(arguments.base_dir + '/data/' + arguments.genotype_matrix.split('.')[0] + '-'
                   + arguments.phenotype_matrix.split('.')[0] + '-' + arguments.phenotype + '.h5', 'a') as f:
        # check if group 'matched_data' and all datasets in 'matched_data' are available, if not: raise Exception
        if 'matched_data' not in f:
            raise Exception('matched_data not in index file ' + arguments.genotype_matrix.split('.')[0] + '-'
                            + arguments.phenotype_matrix.split('.')[0] + '-' + arguments.phenotype + '.h5' +
                            '. See documentation for help.')
        if not all(z in f['matched_data'] for z in matched_datasets):
            raise Exception('Crucial datasets missing in matched_data in index file '
                            + arguments.genotype_matrix.split('.')[0] + '-' + arguments.phenotype_matrix.split('.')[0] +
                            '-' + arguments.phenotype + '.h5' + '. See documentation for help.')
        # check if group 'maf_filter' is available and if user input maf is available, if not: create group/dataset
        if 'maf_filter' not in f:
            maf = f.create_group('maf_filter')
            tmp = (create_maf_filter(arguments.maf_percentage, f['matched_data/ma_frequency']))
            maf.create_dataset(f'maf_{arguments.maf_percentage}', data=tmp, chunks=True, compression="gzip")
        elif f'maf_{arguments.maf_percentage}' not in f['maf_filter']:
            tmp = (create_maf_filter(arguments.maf_percentage, f['matched_data/ma_frequency']))
            f.create_dataset(f'maf_filter/maf_{arguments.maf_percentage}', data=tmp, chunks=True, compression="gzip")
        # check if group datasplit and all user inputs concerning datasplits are available, if not: create all
        if arguments.datasplit == 'nested-cv':
            if 'datasplits' not in f or ('datasplits' in f and 'nested-cv' not in f['datasplits']) or \
                    ('datasplits' in f and 'nested-cv' in f['datasplits'] and f'{arguments.n_outerfolds}-'
                        f'{arguments.n_innerfolds}' not in f['datasplits/nested-cv']):
                nest = f.create_group(f'datasplits/nested-cv/{arguments.n_outerfolds}-{arguments.n_innerfolds}')
                for outer in range(arguments.n_outerfolds):
                    index_dict = check_train_test_splits('nested-cv', f['matched_data/y'],
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
                    ('datasplits' in f and 'cv-test' in f['datasplits'] and f'{arguments.n_innerfolds}-'
                        f'{arguments.test_set_size_percentage}' not in f['datasplits/cv-test']):
                cv = f.create_group(f'datasplits/cv-test/{arguments.n_innerfolds}-{arguments.test_set_size_percentage}')
                index_dict, test = check_train_test_splits('cv-test', f['matched_data/y'],
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
                     f'{100 - arguments.validation_set_size_percentage - arguments.test_set_size_percentage}-'
                     f'{arguments.validation_set_size_percentage}-{arguments.test_set_size_percentage}' not in
                     f['datasplits/cv-test']):
                tvt = f.create_group(f'datasplits/train-val-test/'
                        f'{100 - arguments.validation_set_size_percentage - arguments.test_set_size_percentage}-'
                            f'{arguments.validation_set_size_percentage}-{arguments.test_set_size_percentage}')
                train, val, test = check_train_test_splits('train-val-test', f['matched_data/y'],
                                    [arguments.validation_set_size_percentage, arguments.test_set_size_percentage])
                o = tvt.create_group('outerfold_0')
                o.create_dataset('test', data=test, chunks=True, compression="gzip")
                i = o.create_group('innerfold_0')
                i.create_dataset('train', data=train, chunks=True, compression="gzip")
                i.create_dataset('val', data=val, chunks=True, compression="gzip")


def create_index_file(arguments: argparse.Namespace, X: np.array, y: np.array,  sample_ids: np.array, X_index: np.array,
                      y_index: np.array):
    """
    Function to create .h5 index file described in check_create_index_file().
    :param arguments: all arguments specified by the user
    :param X: genotype in additive encoding to create ma-frequencies
    :param y: matched phenotype values
    :param sample_ids: matched sample ids of genotype/phenotype
    :param X_index: index file of genotype to redo matching
    :param y_index: index file of phenotype to redo matching
    :return:
    """
    freq = get_minor_allele_freq(X)
    maf_threshold = [1, 3, 5]  # standard values for maf threshold
    if arguments.maf_percentage not in maf_threshold:  # add user input if needed
        maf_threshold.append(arguments.maf_percentage)
    param_nested = [[5, 5]]  # standard values for outer and inner folds for nested-cv
    param_cv = [[5, 20]]  # standard number of folds and test percentage for cv-test
    param_tvt = [[20, 20]]  # standard train and val percentages for train-val-test split
    param_nested = check_datasplit_user_input(arguments, 'nested-cv', param_nested)
    param_cv = check_datasplit_user_input(arguments, 'cv-test', param_cv)
    param_tvt = check_datasplit_user_input(arguments, 'train-val-test', param_tvt)

    with h5py.File(arguments.base_dir + '/data/' + arguments.genotype_matrix.split('.')[0] + '-'
                   + arguments.phenotype_matrix.split('.')[0] + '-' + arguments.phenotype + '.h5', 'w') as f:
        # all data needed to redo matching of X and y and to create new mafs and new data splits
        data = f.create_group('matched_data')
        data.create_dataset('y', data=y, chunks=True, compression="gzip")
        data.create_dataset('matched_sample_ids', data=sample_ids, chunks=True, compression="gzip")
        data.create_dataset('X_index', data=X_index, chunks=True, compression="gzip")
        data.create_dataset('y_index', data=y_index, chunks=True, compression="gzip")
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
            n = nest.create_group(f'{elem[0]}-{elem[1]}')
            for outer in range(elem[0]):
                index_dict = check_train_test_splits('nested-cv', y, elem)
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
            index_dict, test = check_train_test_splits('cv-test', y, elem)
            n = cv.create_group(f'{elem[0]}-{elem[1]}')
            o = n.create_group('outerfold_0')
            o.create_dataset('test', data=test, chunks=True, compression="gzip")
            for fold in range(elem[0]):
                i = o.create_group(f'innerfold_{fold}')
                i.create_dataset('train', data=index_dict[f'fold_{fold}_train'], chunks=True, compression="gzip")
                i.create_dataset('val', data=index_dict[f'fold_{fold}_test'], chunks=True, compression="gzip")
        tvt = dsplit.create_group('train-val-test')
        for elem in param_tvt:
            train, val, test = check_train_test_splits('train-val-test', y, elem)
            n = tvt.create_group(f'{100 - elem[0] - elem[1]}-{elem[0]}-{elem[1]}')
            o = n.create_group('outerfold_0')
            o.create_dataset('test', data=test, chunks=True, compression="gzip")
            i = o.create_group('innerfold_0')
            i.create_dataset('train', data=train, chunks=True, compression="gzip")
            i.create_dataset('val', data=val, chunks=True, compression="gzip")


def get_minor_allele_freq(X: np.array):
    """
    Function to compute minor allele frequencies of genotype matrix
    :param X: genotype matrix in additive encoding
    :return: array with frequencies
    """
    return (np.sum(X, 0)) / (2 * X.shape[0])


def create_maf_filter(maf: int, freq: np.array):
    """
    Function to compute minor allele frequency filter
    :param maf: maf threshold as percentage value
    :param freq: array containing minor allele frequencies
    :return: array containing indices of SNPs with MAF smaller than specified threshold, i.e. SNPs to delete
    """
    return np.where(freq <= maf / 100)[0]


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


def check_train_test_splits(split: str, y: np.array, param: list):
    """
    Function to create stratified train-test splits. Continuous values will be grouped into bins and stratified
    according to those.
    :param split: type of data split ('nested-cv', 'cv-test', 'train-val-test')
    :param y: array with phenotypic values for stratification
    :param param: parameters to use for split
    :return: index arrays for splits
    """
    y_binned = make_bins(y)
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


def make_bins(y: np.array):
    """
    Function to create bins of continuous values for stratification.
    :param y: array containing phenotypic values
    :return: binned array
    """
    if test_likely_categorical(y):
        return y
    else:
    # TODO check for number of samples in bins --> join bins if not enough
        _, edges = np.histogram(y)
        edges = edges[:-1]
        y_binned = np.digitize(y, edges)
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
        index_dict[f'fold_{fold}_train'] = x[train_index]
        index_dict[f'fold_{fold}_test'] = x[test_index]
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
    # TODO check for number of samples in test --> error if not enough
    x = np.arange(len(y))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size/100, stratify=y, random_state=random)
    if not val:
        return x_train, x_test, y_train
    else:
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size/100, stratify=y_train,
                                                          random_state=random)
        return x_train, x_val, x_test
