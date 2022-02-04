import argparse
import pandas as pd
import numpy as np
import h5py
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


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
                    'nuc': nucleotide (ATCG)
                    'onehot': one-hot
    optional:   genotype in additional encodings
    :param arguments: all arguments specified by the user
    :return: sample_ids, X_012
    """
    suffix = arguments.genotype_matrix.split('.')[-1]
    encoding = [] # TODO Liste aller Codierungen die überprüft werden müssen
    if suffix in ('h5', 'hdf5', 'h5py'):
        sample_ids, X_012 = check_genotype_h5_file(arguments.genotype_matrix, encoding)
    else:
        if suffix == 'csv':
            sample_ids, snp_ids, X_012, X_nuc = check_genotype_csv_file(arguments.genotype_matrix, encoding)

        elif suffix in ('bed', 'bim', 'fam'):
            sample_ids, snp_ids, X_012, X_nuc = check_genotype_binary_plink_file(arguments.genotype_matrix)

        elif suffix in ('map', 'ped'):
            sample_ids, snp_ids, X_012, X_nuc = check_genotype_plink_file(arguments.genotype_matrix)
        else:
            raise Exception('Only accept .h5, .hdf5, .h5py, .csv, binary PLINK and PLINK genotype files. '
                            'See documentation for help.')
        create_genotype_h5_file(arguments, sample_ids, snp_ids, X_012, X_nuc)
    return X_012, sample_ids


def check_genotype_h5_file(gt_file: str, encoding: list):
    """
    Function to load and check .h5 genotype file. Should contain:
    sample_ids: vector with sample names of genotype matrix,
    snp_ids: vector with SNP identifiers of genotype matrix,
    X_{enc}: (samples x SNPs)-genotype matrix in enc encoding, where enc might refer to:
            '012': additive (number of minor alleles)
            'nuc': nucleotide (ATCG)
    In order to work with other encodings X_nuc is required.
    :param gt_file: path to genotype file
    :param encoding: list of needed encodings
    :return: sample_ids, X_012
    """
    with h5py.File(gt_file, "r") as f:
        keys = list(f.keys())
        if {'sample_ids', 'SNP_ids'}.issubset(keys):
            sample_ids = f['sample_ids'][:]
            # snp_ids = f['snp_ids'][:]
            if 'X_nuc' in keys:
                if 'X_012' in keys:
                    X_012 = f['X_012'][:]
                else:
                    X_nuc = f['X_nuc'][:]
                    X_012 = encode_raw_genotype(X_nuc, '012')
            elif any(z in encoding for z in ['nuc', 'onehot']):  # TODO adapt when we have additional encodings
                raise Exception('Genotype in nucleotide encoding missing in ' + gt_file +
                                '. Can not create required encodings. See documentation for help.')
            elif 'X_012' in keys:
                X_012 = f['X_012'][:]
            else:
                raise Exception('No genotype matrix in file ' + gt_file + '. Need genotype matrix in nucleotide or'
                                                                          'additive encoding. '
                                                                          'See documentation for help')
        else:
            raise Exception('keys "sample_ids" and/or "SNP_ids" are missing in' + gt_file)
    return sample_ids, X_012


def check_genotype_csv_file(gt_file: str, encoding: list):
    """
    Function to load .csv genotype file. File must have the following structure:
    First column must contain the sample ids, the column names should be the SNP ids.
    The values should be the genotype matrix either in additive encoding or in nucleotide encoding.
    If the genotype is in nucleotide encoding, additive and one hot encoding will be calculated.
    If genotype is in additive encoding, only this encoding will be returned.
    :param gt_file: path to genotype file
    :param encoding: list of needed encodings
    :return: sample ids, SNP ids and genotype in additive and nucleotide encoding (if available)
    """
    gt = pd.read_csv(gt_file, index_col=0)
    snp_ids = np.asarray(gt.columns.values)
    sample_ids = np.asarray(gt.index)
    X = np.asarray(gt.values)
    # check encoding of X, only accept additive or nucleotides
    unique = np.unique(X)
    if all(z in ['A', 'C', 'G', 'T'] for z in unique):  # TODO heterozygous!?
        X_012 = encode_raw_genotype(X, '012')
        return sample_ids, snp_ids, X_012, X
    elif all(z in [0, 1, 2] for z in unique):
        if any(z in encoding for z in ['nuc', 'onehot']):
            raise Exception('Genotype in ' + gt_file + ' is not in nucleotide encoding. Can not create required '
                                                       'encodings. See documentation for help.')
        return sample_ids, snp_ids, X, None
    else:
        raise Exception('Genotype in ' + gt_file + ' is neither in additive nor in nucleotide encoding. '
                                                   'See documentation for help.')


def check_genotype_binary_plink_file(gt_file: str):
    """
    Function to load binary PLINK file, .bim, .fam, .bed files with same prefix need to be in same folder.
    Compute additive, nucleotide and one-hot encoding of genotype.
    :param gt_file: path to genotype file
    :return: sample ids, SNP ids and genotype in additive and nucleotide encoding
    """
    gt = read_plink1_bin(gt_file, ref="a0", verbose=False)
    sample_ids = np.array(gt['fid'], dtype=np.int).flatten()
    snp_ids = np.array(gt['snp']).flatten()
    # get nucleotide encoding
    a = np.stack(
        (np.array(gt.a1.values, dtype='S1'), np.zeros(gt.a0.shape, dtype='S1'), np.array(gt.a0.values, dtype='S1')))
    col = np.arange(len(a[0]))
    X_012 = np.array(gt.values)
    X_nuc = a[X_012.astype(int), col]
    return sample_ids, snp_ids, X_012, X_nuc


def check_genotype_plink_file(gt_file: str):
    """
    Function to load PLINK files, .map and .ped file with same prefix need to be in same folder.
    Compute additive, nucleotide and one-hot encoding of genotype.
    :param gt_file: path to genotype file
    :return: sample ids, SNP ids and genotype in additive and nucleotide encoding
    """
    x_file = gt_file.split('.')[0]
    with open(x_file + '.map', 'r') as f:
        SNP_ids = []
        for line in f:
            tmp = line.strip().split(" ")
            SNP_ids.append(tmp[1].strip())
    snp_ids = np.array(SNP_ids)
    iupac_map = {"AA": "A", "GG": "G", "TT": "T", "CC": "C", "AG": "R", "GA": "R", "CT": "Y", "TC": "Y", "GC": "S",
                 "CG": "S", "AT": "W", "TA": "W", "GT": "K", "TG": "K", "AC": "M", "CA": "M"}
    with open(x_file + '.ped', 'r') as f:
        sample_ids = []
        X_nuc = []
        for line in f:
            tmp = line.strip().split(" ")
            sample_ids.append(int(tmp[1].strip()))
            snps = []
            j = 6
            while j < len(tmp) - 1:
                snps.append(iupac_map[tmp[j] + tmp[j + 1]])
                j += 2
            X_nuc.append(snps)
    sample_ids = np.array(sample_ids)
    X_nuc = np.array(X_nuc)
    X_012 = encode_raw_genotype(X_nuc, '012')
    return sample_ids, snp_ids, X_012, X_nuc


def encode_raw_genotype(X: np.array, encoding: str):
    if encoding == '012':
        return get_additive_encoding(X)
    elif encoding == 'onehot':
        return get_onehot_encoding(X)
    else:
        raise Exception('Only able to create additive or one-hot encoding.')


def get_additive_encoding(X: np.array):
    """
    generate genotype matrix in additive encoding:
    0: homozygous major allele,
    1: heterozygous
    2: homozygous minor allele
    :param X: genotype matrix in nucleotide encoding
    :return: X_012
    """
    # TODO heterozygous
    maj_min = []
    index_arr = []
    for col in np.transpose(X):
        _, inv, counts = np.unique(col, return_counts=True, return_inverse=True)
        tmp = np.where(counts == np.max(counts), 0., 2.)
        maj_min.append(tmp)
        index_arr.append(inv)
    maj_min = np.transpose(np.array(maj_min))
    ind_arr = np.transpose(np.array(index_arr))
    cols = np.arange(maj_min.shape[1])
    return maj_min[ind_arr, cols]


def get_onehot_encoding(X: np.array):
    """
    generate genotype matrix in one-hot encoding.
    :param X: genotype matrix in nucleotide encoding
    :return: X_onehot
    """
    # TODO
    raise NotImplementedError


def create_genotype_h5_file(arguments: argparse.Namespace, sample_ids: np.array, snp_ids: np.array, X_012: np.array,
                            X_nuc: np.array):
    """
    Save genotype matrix in unified .h5 file.
    Structure:
                sample_ids
                snp_ids
                X_nuc (or X_012 if X_nuc not available)
    :param arguments:
    :param sample_ids: array containing sample ids of genotype data
    :param snp_ids: array containing snp ids of genotype data
    :param X_012: matrix containing genotype in additive encoding
    :param X_nuc: matrix containing genotype in nucleotide encoding
    """
    with h5py.File(arguments.genotype_matrix.split('.')[0] + '.h5', 'w') as f:
        f.create_dataset('sample_ids', data=sample_ids, chunks=True, compression="gzip")
        f.create_dataset('snp_ids', data=snp_ids, chunks=True, compression="gzip")
        if X_nuc is not None:
            f.create_dataset('X_nuc', data=X_nuc, chunks=True, compression="gzip", compression_opts=7)
        else:
            f.create_dataset('X_012', data=X_012, chunks=True, compression="gzip", compression_opts=7)


def check_and_load_phenotype_matrix(arguments: argparse.Namespace):
    """
    Function to check and load the specified phenotype matrix. Only accept .csv files.
    Column name of sample ids has to be "accession_id", remaining columns should contain phenotypic values
    with phenotype name as column name.
    :param arguments: all arguments specified by the user
    :return: DataFrame with sample_ids as index and phenotype values as single column without NAN values
    """
    if arguments.phenotype_matrix.suffix == ".csv":
        y = pd.read_csv(arguments.phenotype_matrix)
        if 'accession_id' not in y.columns:
            raise Exception('accession_ids are not in phenotype file ' + arguments.phenotype_matrix +
                            '. See documentation for help.')
        else:
            y.sort_values(['accession_id']).groupby('accession_id').mean()
        if arguments.phenotype not in y.columns:
            raise Exception('Phenotype ' + arguments.phenotype + ' is not in phenotype file '
                            + arguments.phenotype_matrix + ' See documentation for help')
        else:
            y = y[[arguments.phenotype]].dropna()
    else:
        raise Exception('Only accept .csv phenotype files. See documentation for help')
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
    if len(y_index[0]) == 0:
        raise Exception('Samples of genotype and phenotype do not match.')
    X = get_matched_data(X, X_index)
    X_ids = get_matched_data(X_ids, X_index)
    y = get_matched_data(y, y_index)
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


def check_create_index_file(arguments: argparse.Namespace, X: np.array, y: np.array, sample_ids: np.array, X_index: np.array,
                            y_index: np.array):
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
    :return:
    """
    filename = arguments.genotype_matrix.split('.')[0] + '-' + arguments.phenotype_matrix.split('.')[0] + '-' + \
               arguments.phenotype + '.h5'
    if os.path.isfile(filename):
        append_index_file(arguments)
    else:
        create_index_file(arguments, X, y, sample_ids, X_index, y_index)


def append_index_file(arguments: argparse.Namespace):
    # TODO
    # TODO wenn file existiert, kann man davon ausgehen, dass matching gemacht wurde
    raise NotImplementedError


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
    if arguments.maf not in maf_threshold:  # add user input if needed
        maf_threshold.append((arguments.maf))
    param_nested = [[5, 5]]  # standard values for outer and inner folds for nested-cv
    param_cv = [[5, 20]]  # standard number of folds and test percentage for cv-test
    param_tvt = [[20, 20]]  # standard train and val percentages for train-val-test split
    param_nested = check_user_input(arguments, 'nested-cv', param_nested)
    param_cv = check_user_input(arguments, 'cv-test', param_cv)
    param_tvt = check_user_input(arguments, 'train-val-test', param_tvt)

    with h5py.File(arguments.genotype_matrix.split('.')[0] + '-' + arguments.phenotype_matrix.split('.')[0] + '-' + \
               arguments.phenotype + '.h5', 'w') as f:
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
        nest = dsplit.create_group('nested-cv')  # TODO anpassen an Änderung in make_stratified_cv!!!
        for elem in param_nested:
            n = nest.create_group(f'{elem[0]}-{elem[1]}')
            for outer in range(elem[0]):
                index_dict = check_train_test_splits('nested-cv', y, elem)
                o = n.create_group(f'outerfold_{outer}')
                o.create_dataset('test', data=index_dict[f'outerfold_{outer}_test'], chunks=True, compression="gzip")
                for inner in range(elem[1]):
                    i = o.create_group(f'innerfold_{inner}')
                    i.create_dataset('train', data=index_dict[f'outerfold_{outer}_inner_train'][inner], chunks=True,
                                     compression="gzip")
                    i.create_dataset('val', data=index_dict[f'outerfold_{outer}_inner_test'][inner], chunks=True,
                                     compression="gzip")
        cv = dsplit.create_group('cv-test')
        for elem in param_cv:
            cv_train, cv_val, test = check_train_test_splits('cv-test', y, elem)
            n = cv.create_group(f'{elem[0]}-{elem[1]}')
            o = n.create_group('outerfold_0')
            o.create_dataset('test', data=test, chunks=True, compression="gzip")
            for fold in range(elem[0]):
                i = o.create_group(f'innerfold_{fold}')
                i.create_dataset('train', data=cv_train[fold], chunks=True, compression="gzip")
                i.create_dataset('val', data=cv_val[fold], chunks=True, compression="gzip")
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
        input = [arguments.outerfolds, arguments.folds]
    elif split == 'cv-test':
        input = [arguments.folds, arguments.testperc]
    elif split == 'train-val-test':
        input = [arguments.valperc, arguments.testperc]
    else:
        raise Exception('Only accept nested-cv, cv-test or train-val-test as data splits.')
    if arguments.datasplit == split and input not in param:
        param.append(input)
        return param

# TODO split Funktionen aufräumen
def check_train_test_splits(split: str, y: np.array, param: list):
    """
    Function to create stratified train-test splits. Continuous values will be grouped into bins and stratified
    according to those.
    :param split: type of data split ('nested-cv', 'cv-test', 'train-val-test')
    :param y: array with phenotypic values for stratification
    :param param: parameters to use for split
    :return: index arrays for splits
    """
    y_binned = make_bins(y)  # TODO discrete vs contin
    if split == 'nested-cv':
        return make_nested_cv(y=y_binned, outer_splits=param[0], inner_splits=param[1])
    elif split == 'cv-test':
        x_train, x_test, y_train = make_train_test_split(y=y_binned, test_size=param[1], val=False)
        cv_train, cv_test = make_stratified_cv(x=x_train, y=y_train, split_number=param[0])
        return cv_train, cv_test, x_test
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
    # TODO check if contin or discrete distribution
    _, edges = np.histogram(y)
    edges = edges[:-1]
    y_binned = np.digitize(y, edges)
    return y_binned


def make_stratified_cv(x, y, split_number):
    """
    Function to create index arrays for stratified cross-validation.
    :param x:
    :param y:
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


def make_train_test_split(y, test_size, val_size=None, val=False, random=42):
    """

    :param y:
    :param test_size:
    :param val_size:
    :param val:
    :param random:
    :return:
    """
    # --> shuffle split, TODO check for number of samples in test --> error if not enough
    x = np.arange(len(y))
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, stratify=y, random_state=random)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random)
    if val == False:
        return x_train, x_test, y_train
    else:
        test_size = test_size / (1 - test_size)
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=test_size, stratify=y_train, random_state=random)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=random)
        return x_train, x_val, x_test


def make_nested_cv(y, outer_splits, inner_splits):
    """
    # TODO anpassen an Änderung in make_stratified_cv!!!
    :param y:
    :param outer_splits:
    :param inner_splits:
    :return:
    """
    outer_cv = StratifiedKFold(n_splits=outer_splits)
    index_dict = {}
    outer_fold = 0
    for train_index, test_index in outer_cv.split(np.zeros(len(y)), y):
        index_dict[f'outerfold_{outer_fold}_test'] = test_index
        inner_train, inner_test = make_strat_cv(train_index, y[train_index], split_number=inner_splits)
        index_dict[f'outerfold_{outer_fold}_inner_train'] = inner_train
        index_dict[f'fold_{outer_fold}_inner_test'] = inner_test
        outer_fold += 1
    return index_dict
