import pandas as pd
import numpy as np
import h5py
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from pandas_plink import read_plink1_bin
from utils import helper_functions
from preprocess import encoding_functions as enc


def prepare_data_files(data_dir: str, genotype_matrix_name: str, phenotype_matrix_name: str, phenotype: str,
                       datasplit: str, n_outerfolds: int, n_innerfolds: int, test_set_size_percentage: int,
                       val_set_size_percentage: int, models, user_encoding: str, maf_percentage: int):
    """
    Prepare all data files for a common format: genotype matrix, phenotype matrix and index file.

    First check if genotype file is .h5 file (standard format of this framework):
        YES:    First check if all required information is present in the file, raise Exception if not.
                Then check if index file exists:
                    NO: Load genotype and create all required index files
                    YES: Append all required data splits and maf-filters to index file
        NO:     Load genotype and create all required files
    :param data_dir: data directory where the phenotype and genotype matrix are stored
    :param genotype_matrix_name: name of the genotype matrix including datatype ending
    :param phenotype_matrix_name: name of the phenotype matrix including datatype ending
    :param phenotype: name of the phenotype to predict
    :param datasplit: datasplit to use. Options are: nested-cv, cv-test, train-val-test
    :param n_outerfolds: number of outerfolds relevant for nested-cv
    :param n_innerfolds: number of folds relevant for nested-cv and cv-test
    :param test_set_size_percentage: size of the test set relevant for cv-test and train-val-test
    :param val_set_size_percentage: size of the validation set relevant for train-val-test
    :param models: models to consider
    :param user_encoding: encoding specified by the user
    :param maf_percentage: threshold for MAF filter as percentage value
    """
    print('Check if all data files have the required format')
    if os.path.isfile(data_dir + '/' + genotype_matrix_name.split('.')[0] + '.h5') and \
            (genotype_matrix_name.split('.')[-1] != 'h5'):
        print("Found same file name with ending .h5")
        print("Assuming that the raw file was already prepared using our pipepline. Will continue with the .h5 file.")
        genotype_matrix_name = genotype_matrix_name.split('.')[0] + '.h5'
    suffix = genotype_matrix_name.split('.')[-1]
    if suffix in ('h5', 'hdf5', 'h5py'):
        # Genotype matrix has standard file format -> check information in the file
        check_genotype_h5_file(data_dir=data_dir, genotype_matrix_name=genotype_matrix_name,
                               encodings=enc.get_encoding(models=models, user_encoding=user_encoding))
        print('Genotype file available in required format, check index file now.')
        # Check / create index files
        if check_index_file(data_dir=data_dir, genotype_matrix_name=genotype_matrix_name,
                            phenotype_matrix_name=phenotype_matrix_name, phenotype=phenotype):
            print('Index file ' + genotype_matrix_name.split('.')[0] + '-'
                    + phenotype_matrix_name.split('.')[0] + '-' + phenotype + '.h5' + ' already exists.'
                    ' Will append required filters and data splits now.')
            append_index_file(data_dir=data_dir, genotype_matrix_name=genotype_matrix_name,
                              phenotype_matrix_name=phenotype_matrix_name, phenotype=phenotype,
                              maf_percentage=maf_percentage,
                              datasplit=datasplit, n_outerfolds=n_outerfolds, n_innerfolds=n_innerfolds,
                              test_set_size_percentage=test_set_size_percentage,
                              val_set_size_percentage=val_set_size_percentage)
            print('Done checking data files. All required datasets are available.')
        else:
            print('Index file ' + genotype_matrix_name.split('.')[0] + '-' + phenotype_matrix_name.split('.')[0]
                  + '-' + phenotype + '.h5' + ' does not fulfill requirements. '
                                              'Will load genotype and phenotype matrix and create new index file.')
            save_all_data_files(data_dir=data_dir, genotype_matrix_name=genotype_matrix_name,
                                phenotype_matrix_name=phenotype_matrix_name, phenotype=phenotype,
                                models=models, user_encoding=user_encoding, maf_percentage=maf_percentage,
                                datasplit=datasplit, n_outerfolds=n_outerfolds, n_innerfolds=n_innerfolds,
                                test_set_size_percentage=test_set_size_percentage,
                                val_set_size_percentage=val_set_size_percentage)
            print('Done checking data files. All required datasets are available.')
    else:
        print('Genotype file not in required format. Will load genotype matrix and save as .h5 file. Will also create '
              'required index file.')
        save_all_data_files(data_dir=data_dir, genotype_matrix_name=genotype_matrix_name,
                            phenotype_matrix_name=phenotype_matrix_name, phenotype=phenotype,
                            models=models, user_encoding=user_encoding, maf_percentage=maf_percentage,
                            datasplit=datasplit, n_outerfolds=n_outerfolds, n_innerfolds=n_innerfolds,
                            test_set_size_percentage=test_set_size_percentage,
                            val_set_size_percentage=val_set_size_percentage)
        print('Done checking data files. All required datasets are available.')


def check_genotype_h5_file(data_dir: str, genotype_matrix_name: str, encodings: list):
    """
    Check .h5 genotype file.
    Should contain:
        sample_ids: vector with sample names of genotype matrix,
        snp_ids: vector with SNP identifiers of genotype matrix,
        X_{enc}: (samples x SNPs)-genotype matrix in enc encoding, where enc might refer to:
                    '012': additive (number of minor alleles)
                    'raw': raw (alleles)
    :param data_dir: data directory where the phenotype and genotype matrix are stored
    :param genotype_matrix_name: name of the phenotype matrix including datatype ending
    :param encodings: list of needed encodings
    """
    with h5py.File(data_dir + '/' + genotype_matrix_name, "r") as f:
        keys = list(f.keys())
        if {'sample_ids', 'snp_ids'}.issubset(keys):
            # check if required encoding is available or can be created
            for elem in encodings:
                if f'X_{elem}' not in f and f'X_{enc.get_base_encoding(encoding=elem)}' not in f:
                    raise Exception('Genotype in ' + elem + ' encoding missing. Can not create required encoding. '
                                                            'See documentation for help')
        else:
            raise Exception('sample_ids and/or snp_ids are missing in' + genotype_matrix_name)


def check_index_file(data_dir: str, genotype_matrix_name: str, phenotype_matrix_name: str, phenotype: str) -> bool:
    """
    Check if index file is available and if the datasets 'y', 'matched_sample_ids', 'X_index', 'y_index' and
    'ma_frequency' exist.
    :param data_dir: data directory where the phenotype and genotype matrix are stored
    :param genotype_matrix_name: name of the genotype matrix including datatype ending
    :param phenotype_matrix_name: name of the phenotype matrix including datatype ending
    :param phenotype: name of the phenotype to predict
    :return: bool reflecting check result
    """
    index_file = data_dir + '/' + genotype_matrix_name.split('.')[0] + '-' + phenotype_matrix_name.split('.')[0] \
                 + '-' + phenotype + '.h5'
    if os.path.isfile(index_file):
        matched_datasets = ['y', 'matched_sample_ids', 'X_index', 'y_index', 'non_informative_filter', 'ma_frequency']
        with h5py.File(index_file, 'a') as f:
            if 'matched_data' in f and all(z in f['matched_data'] for z in matched_datasets):
                return True
            else:
                return False
    else:
        return False


def save_all_data_files(data_dir: str, genotype_matrix_name: str, phenotype_matrix_name: str, phenotype: str,
                        models, user_encoding: str, maf_percentage: int,
                        datasplit: str, n_outerfolds: int, n_innerfolds: int,
                        test_set_size_percentage: int, val_set_size_percentage: int):
    """
    Prepare and save all required data files:
        - genotype matrix in unified format as .h5 file with,
        - phenotype matrix in unified format as .csv file,
        - file containing maf filter and data split indices as .h5
    :param data_dir: data directory where the phenotype and genotype matrix are stored
    :param genotype_matrix_name: name of the genotype matrix including datatype ending
    :param phenotype_matrix_name: name of the phenotype matrix including datatype ending
    :param phenotype: name of the phenotype to predict
    :param models: models to consider
    :param user_encoding: encoding specified by the user
    :param maf_percentage: threshold for MAF filter as percentage value
    :param datasplit: datasplit to use. Options are: nested-cv, cv-test, train-val-test
    :param n_outerfolds: number of outerfolds relevant for nested-cv
    :param n_innerfolds: number of folds relevant for nested-cv and cv-test
    :param test_set_size_percentage: size of the test set relevant for cv-test and train-val-test
    :param val_set_size_percentage: size of the validation set relevant for train-val-test
    """
    print('Load genotype file ' + data_dir + '/' + genotype_matrix_name)
    X, X_ids = check_transform_format_genotype_matrix(data_dir=data_dir, genotype_matrix_name=genotype_matrix_name,
                                                      models=models, user_encoding=user_encoding)
    print('Have genotype matrix. Load phenotype ' + phenotype + ' from ' + data_dir + '/' + phenotype_matrix_name)
    y = check_and_load_phenotype_matrix(data_dir=data_dir,
                                        phenotype_matrix_name=phenotype_matrix_name, phenotype=phenotype)
    print('Have phenotype vector. Start matching genotype and phenotype.')
    X, y, sample_ids, X_index, y_index = genotype_phenotype_matching(X=X, X_ids=X_ids, y=y)
    print('Done matching genotype and phenotype. Create index file now.')
    create_index_file(data_dir=data_dir, genotype_matrix_name=genotype_matrix_name,
                      phenotype_matrix_name=phenotype_matrix_name, phenotype=phenotype,
                      datasplit=datasplit, n_outerfolds=n_outerfolds, n_innerfolds=n_innerfolds,
                      test_set_size_percentage=test_set_size_percentage,
                      val_set_size_percentage=val_set_size_percentage,
                      maf_percentage=maf_percentage, X=X, y=y, sample_ids=sample_ids, X_index=X_index, y_index=y_index
                      )


def check_transform_format_genotype_matrix(data_dir: str, genotype_matrix_name: str, models, user_encoding: str) \
        -> (np.array, np.array):
    """
    Check the format of the specified genotype matrix.
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
    :param data_dir: data directory where the phenotype and genotype matrix are stored
    :param genotype_matrix_name: name of the genotype matrix including datatype ending
    :param models: models to consider
    :param user_encoding: encoding specified by the user
    :return: genotype matrix (raw encoded if present, 012 encoded otherwise) and sample ids
    """
    suffix = genotype_matrix_name.split('.')[-1]
    encoding = enc.get_encoding(models=models, user_encoding=user_encoding)
    if suffix in ('h5', 'hdf5', 'h5py'):
        with h5py.File(data_dir + '/' + genotype_matrix_name, "r") as f:
            sample_ids = f['sample_ids'][:].astype(str)
            if 'X_raw' in f:
                X = f['X_raw'][:]
            elif 'X_012' in f:
                X = f['X_012'][:]
    else:
        if suffix == 'csv':
            sample_ids, snp_ids, X = check_genotype_csv_file(data_dir=data_dir,
                                                             genotype_matrix_name=genotype_matrix_name,
                                                             encodings=encoding)

        elif suffix in ('bed', 'bim', 'fam'):
            sample_ids, snp_ids, X = check_genotype_binary_plink_file(data_dir=data_dir,
                                                                      genotype_matrix_name=genotype_matrix_name)

        elif suffix in ('map', 'ped'):
            sample_ids, snp_ids, X = check_genotype_plink_file(data_dir=data_dir,
                                                               genotype_matrix_name=genotype_matrix_name)
        else:
            raise Exception('Only accept .h5, .hdf5, .h5py, .csv, binary PLINK and PLINK genotype files. '
                            'See documentation for help.')
        create_genotype_h5_file(data_dir=data_dir, genotype_matrix_name=genotype_matrix_name,
                                sample_ids=sample_ids, snp_ids=snp_ids, X=X)
    return X, sample_ids


def check_genotype_csv_file(data_dir: str, genotype_matrix_name: str, encodings: list) \
        -> (np.array, np.array, np.array):
    """
    Load .csv genotype file. File must have the following structure:
    First column must contain the sample ids, the column names should be the SNP ids.
    The values should be the genotype matrix either in additive encoding or in raw encoding.
    If the genotype is in raw encoding, additive encoding will be calculated.
    If genotype is in additive encoding, only this encoding will be returned
    :param data_dir: data directory where the phenotype and genotype matrix are stored
    :param genotype_matrix_name: name of the genotype matrix including datatype ending
    :param encodings: list of needed encodings
    :return: sample ids, SNP ids and genotype in additive / raw encoding (if available)
    """
    gt = pd.read_csv(data_dir + '/' + genotype_matrix_name, index_col=0)
    snp_ids = np.asarray(gt.columns.values)
    sample_ids = np.asarray(gt.index)
    X = np.asarray(gt.values)
    # check encoding of X, only accept additive or raw and check if required encoding can be created
    enc_of_X = enc.check_encoding_of_genotype(X=X)
    for elem in encodings:
        if elem != enc_of_X and enc.get_base_encoding(encoding=elem) != enc_of_X:
            raise Exception('Genotype in ' + genotype_matrix_name + ' in wrong encoding. Can not create'
                            ' required encoding. See documentation for help.')
    return sample_ids, snp_ids, X


def check_genotype_binary_plink_file(data_dir: str, genotype_matrix_name: str) -> (np.array, np.array, np.array):
    """
    Load binary PLINK file, .bim, .fam, .bed files with same prefix need to be in same folder.
    Compute additive and raw encoding of genotype
    :param data_dir: data directory where the phenotype and genotype matrix are stored
    :param genotype_matrix_name: name of the genotype matrix including datatype ending
    :return: sample ids, SNP ids and genotype in raw encoding
    """
    gt_file = data_dir + '/' + genotype_matrix_name.split(".")[0]
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


def check_genotype_plink_file(data_dir: str, genotype_matrix_name: str) -> (np.array, np.array, np.array):
    """
    Load PLINK files, .map and .ped file with same prefix need to be in same folder.
    Accepts GENOTYPENAME.ped and GENOTYPENAME.map as input
    :param data_dir: data directory where the phenotype and genotype matrix are stored
    :param genotype_matrix_name: name of the genotype matrix including datatype ending
    :return: sample ids, SNP ids and genotype in raw encoding
    """
    gt_file = data_dir + '/' + genotype_matrix_name.split(".")[0]
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
    return sample_ids, snp_ids, X_raw


def create_genotype_h5_file(data_dir: str, genotype_matrix_name: str,
                            sample_ids: np.array, snp_ids: np.array, X: np.array):
    """
    Save genotype matrix in unified .h5 file.
    Structure:
                sample_ids
                snp_ids
                X_raw (or X_012 if X_raw not available)
    :param data_dir: data directory where the phenotype and genotype matrix are stored
    :param genotype_matrix_name: name of the genotype matrix including datatype ending
    :param sample_ids: array containing sample ids of genotype data
    :param snp_ids: array containing snp ids of genotype data
    :param X: matrix containing genotype either in raw or in additive encoding
    """
    x_file = data_dir + '/' + genotype_matrix_name.split(".")[0] + '.h5'
    print('Save unified genotype file ' + x_file)
    with h5py.File(x_file, 'w') as f:
        f.create_dataset('sample_ids', data=sample_ids.astype(bytes), chunks=True, compression="gzip")
        f.create_dataset('snp_ids', data=snp_ids.astype(bytes), chunks=True, compression="gzip")
        encoding = enc.check_encoding_of_genotype(X=X)
        if encoding == 'raw':
            f.create_dataset('X_raw', data=X, chunks=True, compression="gzip", compression_opts=7)
        elif encoding == '012':
            f.create_dataset('X_012', data=X, chunks=True, compression="gzip", compression_opts=7)
        else:
            raise Exception('Genotype neither in raw or additive encoding. Cannot save .h5 genotype file.')


def check_and_load_phenotype_matrix(data_dir: str, phenotype_matrix_name: str, phenotype: str) -> pd.DataFrame:
    """
    Check and load the specified phenotype matrix. Only accept .csv, .pheno, .txt files.
    Sample ids need to be in first column, remaining columns should contain phenotypic values
    with phenotype name as column name
    :param data_dir: data directory where the phenotype and genotype matrix are stored
    :param phenotype_matrix_name: name of the phenotype matrix including datatype ending
    :param phenotype: name of the phenotype to predict
    :return: DataFrame with sample_ids as index and phenotype values as single column without NAN values
    """
    suffix = phenotype_matrix_name.split('.')[-1]
    if suffix == "csv":
        y = pd.read_csv(data_dir + '/' + phenotype_matrix_name)
    elif suffix in ("pheno", "txt"):
        y = pd.read_csv(data_dir + '/' + phenotype_matrix_name, sep=" ")
    else:
        raise Exception('Only accept .csv, .pheno, .txt phenotype files. See documentation for help')
    y = y.sort_values(y.columns[0]).groupby(y.columns[0]).mean()
    if phenotype not in y.columns:
        raise Exception('Phenotype ' + phenotype + ' is not in phenotype file ' + phenotype_matrix_name +
                        ' See documentation for help')
    else:
        y = y[[phenotype]].dropna()
    return y


def genotype_phenotype_matching(X: np.array, X_ids: np.array, y: pd.DataFrame) -> tuple:
    """
    Match the handed over genotype and phenotype matrix for the phenotype specified by the user
    :param X: genotype matrix in additive encoding
    :param X_ids: sample ids of genotype matrix
    :param y: pd.DataFrame containing sample ids of phenotype as index and phenotype values as single column
    :return: matched genotype matrix, matched sample ids, index arrays for genotype and phenotype to redo matching
    """
    y_ids = np.asarray(y.index, dtype=X_ids.dtype).flatten()
    (y_index, X_index) = (np.reshape(y_ids, (y_ids.shape[0], 1)) == X_ids).nonzero()
    if len(y_index) == 0:
        raise Exception('Samples of genotype and phenotype do not match.')
    X = get_matched_data(X, X_index)
    X_ids = get_matched_data(X_ids, X_index)
    y = get_matched_data(y.values.flatten(), y_index)
    return X, y, X_ids, X_index, y_index


def get_matched_data(data: np.array, index: np.array) -> np.array:
    """
    Get elements of data specified in index array
    :param data: matrix or array
    :param index: index array
    :return: data at selected indices
    """
    if data.ndim == 2:
        return data[index, :]
    else:
        return data[index]


def append_index_file(data_dir: str, genotype_matrix_name: str, phenotype_matrix_name: str, phenotype: str,
                      datasplit: str, n_outerfolds: int, n_innerfolds: int, test_set_size_percentage: int,
                      val_set_size_percentage: int, maf_percentage: int):
    """
    Check index file, described in create_index_file(), and append datasets if necessary
    :param data_dir: data directory where the phenotype and genotype matrix are stored
    :param genotype_matrix_name: name of the genotype matrix including datatype ending
    :param phenotype_matrix_name: name of the phenotype matrix including datatype ending
    :param phenotype: name of the phenotype to predict
    :param maf_percentage: threshold for MAF filter as percentage value
    :param datasplit: datasplit to use. Options are: nested-cv, cv-test, train-val-test
    :param n_outerfolds: number of outerfolds relevant for nested-cv
    :param n_innerfolds: number of folds relevant for nested-cv and cv-test
    :param test_set_size_percentage: size of the test set relevant for cv-test and train-val-test
    :param val_set_size_percentage: size of the validation set relevant for train-val-test
    """
    with h5py.File(data_dir + '/' + genotype_matrix_name.split('.')[0] + '-'
                   + phenotype_matrix_name.split('.')[0] + '-' + phenotype + '.h5', 'a') as f:
        # check if group 'maf_filter' is available and if user input maf is available, if not: create group/dataset
        if 'maf_filter' not in f:
            maf = f.create_group('maf_filter')
            tmp = (create_maf_filter(maf=maf_percentage, freq=f['matched_data/ma_frequency'][:]))
            maf.create_dataset(f'maf_{maf_percentage}', data=tmp, chunks=True, compression="gzip")
        elif f'maf_{maf_percentage}' not in f['maf_filter']:
            tmp = (create_maf_filter(maf=maf_percentage, freq=f['matched_data/ma_frequency'][:]))
            f.create_dataset(f'maf_filter/maf_{maf_percentage}', data=tmp, chunks=True, compression="gzip")
        # check if group datasplit and all user inputs concerning datasplits are available, if not: create all
        if datasplit == 'nested-cv':
            subpath = helper_functions.get_subpath_for_datasplit(datasplit=datasplit,
                                                                 datasplit_params=[n_outerfolds, n_innerfolds])
            if 'datasplits' not in f or ('datasplits' in f and 'nested-cv' not in f['datasplits']) or \
                    ('datasplits' in f and 'nested-cv' in f['datasplits'] and
                     f'{subpath}' not in f['datasplits/nested-cv']):
                nest = f.create_group(f'datasplits/nested-cv/{subpath}')
                for outer in range(n_outerfolds):
                    index_dict = check_train_test_splits(y=f['matched_data/y'], datasplit='nested-cv',
                                                         datasplit_params=[n_outerfolds, n_innerfolds])
                    o = nest.create_group(f'outerfold_{outer}')
                    o.create_dataset('test', data=index_dict[f'outerfold_{outer}_test'], chunks=True,
                                     compression="gzip")
                    for inner in range(n_innerfolds):
                        i = o.create_group(f'innerfold_{inner}')
                        i.create_dataset('train', data=index_dict[f'outerfold_{outer}'][f'fold_{inner}_train'],
                                         chunks=True,
                                         compression="gzip")
                        i.create_dataset('val', data=index_dict[f'outerfold_{outer}'][f'fold_{inner}_test'],
                                         chunks=True,
                                         compression="gzip")
        elif datasplit == 'cv-test':
            subpath = helper_functions.get_subpath_for_datasplit(datasplit=datasplit,
                                                                 datasplit_params=
                                                                 [n_innerfolds, test_set_size_percentage])
            if 'datasplits' not in f or ('datasplits' in f and 'cv-test' not in f['datasplits']) or \
                    ('datasplits' in f and 'cv-test' in f['datasplits'] and
                     f'{subpath}' not in f['datasplits/cv-test']):
                cv = f.create_group(f'datasplits/cv-test/{subpath}')
                index_dict, test = check_train_test_splits(y=f['matched_data/y'], datasplit='cv-test',
                                                           datasplit_params=[n_innerfolds, test_set_size_percentage])
                o = cv.create_group('outerfold_0')
                o.create_dataset('test', data=test, chunks=True, compression="gzip")
                for fold in range(n_innerfolds):
                    i = o.create_group(f'innerfold_{fold}')
                    i.create_dataset('train', data=index_dict[f'fold_{fold}_train'], chunks=True, compression="gzip")
                    i.create_dataset('val', data=index_dict[f'fold_{fold}_test'], chunks=True, compression="gzip")
        elif datasplit == 'train-val-test':
            subpath = helper_functions.get_subpath_for_datasplit(datasplit=datasplit,
                                                                 datasplit_params=[val_set_size_percentage,
                                                                                   test_set_size_percentage])
            if 'datasplits' not in f or ('datasplits' in f and 'train-val-test' not in f['datasplits']) or \
                    ('datasplits' in f and 'train-val-test' in f['datasplits'] and
                     f'{subpath}' not in f['datasplits/cv-test']):
                tvt = f.create_group(f'datasplits/train-val-test/{subpath}')
                train, val, test = check_train_test_splits(y=f['matched_data/y'], datasplit='train-val-test',
                                                           datasplit_params=[val_set_size_percentage,
                                                                             test_set_size_percentage])
                o = tvt.create_group('outerfold_0')
                o.create_dataset('test', data=test, chunks=True, compression="gzip")
                i = o.create_group('innerfold_0')
                i.create_dataset('train', data=train, chunks=True, compression="gzip")
                i.create_dataset('val', data=val, chunks=True, compression="gzip")


def create_index_file(data_dir: str, genotype_matrix_name: str, phenotype_matrix_name: str, phenotype: str,
                      datasplit: str, n_outerfolds: int, n_innerfolds: int, test_set_size_percentage: int,
                      val_set_size_percentage: int, maf_percentage: int,
                      X: np.array, y: np.array,  sample_ids: np.array, X_index: np.array, y_index: np.array):
    """
    Create the .h5 index file containing the maf filters and data splits for the combination of genotype matrix,
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
    :param data_dir: data directory where the phenotype and genotype matrix are stored
    :param genotype_matrix_name: name of the genotype matrix including datatype ending
    :param phenotype_matrix_name: name of the phenotype matrix including datatype ending
    :param phenotype: name of the phenotype to predict
    :param maf_percentage: threshold for MAF filter as percentage value
    :param datasplit: datasplit to use. Options are: nested-cv, cv-test, train-val-test
    :param n_outerfolds: number of outerfolds relevant for nested-cv
    :param n_innerfolds: number of folds relevant for nested-cv and cv-test
    :param test_set_size_percentage: size of the test set relevant for cv-test and train-val-test
    :param val_set_size_percentage: size of the validation set relevant for train-val-test
    :param X: genotype in additive encoding to create ma-frequencies
    :param y: matched phenotype values
    :param sample_ids: matched sample ids of genotype/phenotype
    :param X_index: index file of genotype to redo matching
    :param y_index: index file of phenotype to redo matching
    """
    X, non_informative_filter = filter_non_informative_snps(X=X)
    freq = get_minor_allele_freq(X=X)
    maf_threshold = [0, 1, 3, 5]  # standard values for maf threshold
    if maf_percentage not in maf_threshold:  # add user input if needed
        maf_threshold.append(maf_percentage)
    param_nested = [[5, 5]]  # standard values for outer and inner folds for nested-cv
    param_cv = [[5, 20]]  # standard number of folds and test percentage for cv-test
    param_tvt = [[20, 20]]  # standard train and val percentages for train-val-test split
    param_nested = check_datasplit_user_input(user_datasplit=datasplit,
                                              user_n_outerfolds=n_outerfolds, user_n_innerfolds=n_innerfolds,
                                              user_test_set_size_percentage=test_set_size_percentage,
                                              user_val_set_size_percentage=val_set_size_percentage,
                                              datasplit='nested-cv', param_to_check=param_nested)
    param_cv = check_datasplit_user_input(user_datasplit=datasplit,
                                          user_n_outerfolds=n_outerfolds, user_n_innerfolds=n_innerfolds,
                                          user_test_set_size_percentage=test_set_size_percentage,
                                          user_val_set_size_percentage=val_set_size_percentage,
                                          datasplit='cv-test', param_to_check=param_cv)
    param_tvt = check_datasplit_user_input(user_datasplit=datasplit,
                                           user_n_outerfolds=n_outerfolds, user_n_innerfolds=n_innerfolds,
                                           user_test_set_size_percentage=test_set_size_percentage,
                                           user_val_set_size_percentage=val_set_size_percentage,
                                           datasplit='train-val-test', param_to_check=param_tvt)

    with h5py.File(data_dir + '/' + genotype_matrix_name.split('.')[0] + '-'
                   + phenotype_matrix_name.split('.')[0] + '-' + phenotype + '.h5', 'w') as f:
        # all data needed to redo matching of X and y and to create new mafs and new data splits
        data = f.create_group('matched_data')
        data.create_dataset('y', data=y, chunks=True, compression="gzip")
        data.create_dataset('matched_sample_ids', data=sample_ids.astype(bytes), chunks=True, compression="gzip")
        data.create_dataset('X_index', data=X_index, chunks=True, compression="gzip")
        data.create_dataset('y_index', data=y_index, chunks=True, compression="gzip")
        data.create_dataset('non_informative_filter', data=non_informative_filter, chunks=True, compression="gzip")
        data.create_dataset('ma_frequency', data=freq, chunks=True, compression="gzip")
        # create and save standard mafs and maf according to user input
        maf = f.create_group('maf_filter')
        for threshold in maf_threshold:
            tmp = (create_maf_filter(maf=threshold, freq=freq))
            maf.create_dataset(f'maf_{threshold}', data=tmp, chunks=True, compression="gzip")
        # create and save standard data splits and splits according to user input
        dsplit = f.create_group('datasplits')
        nest = dsplit.create_group('nested-cv')
        for elem in param_nested:
            n = nest.create_group(helper_functions.get_subpath_for_datasplit(datasplit='nested-cv',
                                                                             datasplit_params=elem))
            for outer in range(elem[0]):
                index_dict = check_train_test_splits(y=y, datasplit='nested-cv', datasplit_params=elem)
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
            index_dict, test = check_train_test_splits(y=y, datasplit='cv-test', datasplit_params=elem)
            n = cv.create_group(helper_functions.get_subpath_for_datasplit(datasplit='cv-test', datasplit_params=elem))
            o = n.create_group('outerfold_0')
            o.create_dataset('test', data=test, chunks=True, compression="gzip")
            for fold in range(elem[0]):
                i = o.create_group(f'innerfold_{fold}')
                i.create_dataset('train', data=index_dict[f'fold_{fold}_train'], chunks=True, compression="gzip")
                i.create_dataset('val', data=index_dict[f'fold_{fold}_test'], chunks=True, compression="gzip")
        tvt = dsplit.create_group('train-val-test')
        for elem in param_tvt:
            train, val, test = check_train_test_splits(y=y, datasplit='train-val-test', datasplit_params=elem)
            n = tvt.create_group(helper_functions.get_subpath_for_datasplit(datasplit='train-val-test',
                                                                            datasplit_params=elem))
            o = n.create_group('outerfold_0')
            o.create_dataset('test', data=test, chunks=True, compression="gzip")
            i = o.create_group('innerfold_0')
            i.create_dataset('train', data=train, chunks=True, compression="gzip")
            i.create_dataset('val', data=val, chunks=True, compression="gzip")


def filter_non_informative_snps(X: np.array) -> (np.array, np.array):
    """
    Remove non-informative SNPs, i.e. SNPs that are constant
    :param X: genotype matrix in raw or additive encoding
    :return: filtered genotype matrix and filter-vector
    """
    tmp = X == X[0, :]
    X_filtered = X[:, ~tmp.all(0)]
    return X_filtered, (~tmp.all(0)).nonzero()[0]


def get_minor_allele_freq(X: np.array) -> np.array:
    """
    Compute minor allele frequencies of genotype matrix
    :param X: genotype matrix in additive encoding
    :return: array with frequencies
    """
    encoding = enc.check_encoding_of_genotype(X=X)
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


def create_maf_filter(maf: int, freq: np.array) -> np.array:
    """
    Create minor allele frequency filter
    :param maf: maf threshold as percentage value
    :param freq: array containing minor allele frequencies as decimal value
    :return: array containing indices of SNPs with MAF smaller than specified threshold, i.e. SNPs to delete
    """
    return np.where(freq <= (maf / 100))[0]


def check_datasplit_user_input(user_datasplit: str, user_n_outerfolds: int, user_n_innerfolds: int,
                               user_test_set_size_percentage: int, user_val_set_size_percentage: int,
                               datasplit: str, param_to_check: list) -> list:
    """
    Check if user input of data split parameters differs from standard values.
    If it does, add input to list of parameters
    :param user_datasplit: datasplit specified by the user
    :param user_n_outerfolds: number of outerfolds relevant for nested-cv specified by the user
    :param user_n_innerfolds: number of folds relevant for nested-cv and cv-test specified by the user
    :param user_test_set_size_percentage:
        size of the test set relevant for cv-test and train-val-test specified by the user
    :param user_val_set_size_percentage: size of the validation set relevant for train-val-test specified by the user
    :param datasplit: type of data split
    :param param_to_check: standard parameters to compare to
    :return: adapted list of parameters
    """
    if datasplit == 'nested-cv':
        user_input = [user_n_outerfolds, user_n_innerfolds]
    elif datasplit == 'cv-test':
        user_input = [user_n_innerfolds, user_test_set_size_percentage]
    elif datasplit == 'train-val-test':
        user_input = [user_val_set_size_percentage, user_test_set_size_percentage]
    else:
        raise Exception('Only accept nested-cv, cv-test or train-val-test as data splits.')
    if user_datasplit == datasplit and user_input not in param_to_check:
        param_to_check.append(user_input)
    return param_to_check


def check_train_test_splits(y: np.array, datasplit: str, datasplit_params: list):
    """
    Create stratified train-test splits. Continuous values will be grouped into bins and stratified according to those
    :param datasplit: type of datasplit ('nested-cv', 'cv-test', 'train-val-test')
    :param y: array with phenotypic values for stratification
    :param datasplit_params: parameters to use for split:
        [n_outerfolds, n_innerfolds] for nested-cv
        [n_innerfolds, test_set_size_percentage] for cv-test
        [val_set_size_percentage, test_set_size_percentage] for train-val-test
    :return: dictionary respectively arrays with indices
    """
    y_binned = make_bins(y=y, datasplit=datasplit, datasplit_params=datasplit_params)
    if datasplit == 'nested-cv':
        return make_nested_cv(y=y_binned, outerfolds=datasplit_params[0], innerfolds=datasplit_params[1])
    elif datasplit == 'cv-test':
        x_train, x_test, y_train = make_train_test_split(y=y_binned, test_size=datasplit_params[1], val=False)
        cv_dict = make_stratified_cv(x=x_train, y=y_train, split_number=datasplit_params[0])
        return cv_dict, x_test
    elif datasplit == 'train-val-test':
        return make_train_test_split(y=y_binned, test_size=datasplit_params[1], val_size=datasplit_params[0], val=True)
    else:
        raise Exception('Only accept nested-cv, cv-test or train-val-test as data splits.')


def make_bins(y: np.array, datasplit: str, datasplit_params: list) -> np.array:
    """
    Create bins of continuous values for stratification
    :param y: array containing phenotypic values
    :param datasplit: train test split to use
    :param datasplit_params: list of parameters to use:
        [n_outerfolds, n_innerfolds] for nested-cv
        [n_innerfolds, test_set_size_percentage] for cv-test
        [val_set_size_percentage, test_set_size_percentage] for train-val-test
    :return: binned array
    """
    if helper_functions.test_likely_categorical(y):
        return y.astype(int)
    else:
        if datasplit == 'nested-cv':
            tmp = len(y)/(datasplit_params[0] + datasplit_params[1])
        elif datasplit == 'cv-test':
            tmp = len(y)*(1-datasplit_params[1]/100)/datasplit_params[0]
        else:
            tmp = len(y)/10 + 1

        number_of_bins = min(int(tmp) - 1, 10)
        edges = np.percentile(y, np.linspace(0, 100, number_of_bins)[1:])
        y_binned = np.digitize(y, edges, right=True)
        return y_binned


def make_nested_cv(y: np.array, outerfolds: int, innerfolds: int) -> dict:
    """
    Create index dictionary for stratified nested cross validation with the following structure:
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
        index_dict[f'outerfold_{outer_fold}'] = make_stratified_cv(x=train_index, y=y[train_index],
                                                                   split_number=innerfolds)
        outer_fold += 1
    return index_dict


def make_stratified_cv(x: np.array, y: np.array, split_number: int) -> dict:
    """
    Create index dictionary for stratified cross-validation with following structure:
        {fold_0_train: fold_0_train_indices,
        fold_0_test: fold_0_test_indices,
        ...
        fold_n_train: fold_n_train_indices,
        fold_n_test: fold_n_test_indices
        }
    :param x: whole train indices
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


def make_train_test_split(y: np.array, test_size: int, val_size=None, val=False, random=42) \
        -> (np.array, np.array, np.array):
    """
    Create index arrays for stratified train-test, respectively train-val-test splits.
    :param y: target values grouped in bins for stratification
    :param test_size: size of test set as percentage value
    :param val_size: size of validation set as percentage value
    :param val: if True, function returns validation set additionally to train and test set
    :param random: controls shuffling of data
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
