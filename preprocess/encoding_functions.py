import argparse
import numpy as np
import torch
from torch.nn.functional import one_hot
from utils import helper_functions


def get_encoding(arguments:  argparse.Namespace):
    """
    Get a list of all required encodings
    :param arguments:
    :return: list of encodings
    """
    if arguments.models == 'all':
        list_of_encodings = get_list_of_encodings()
    else:
        if arguments.encoding is not None:
            list_of_encodings = [arguments.encoding]
        else:
            list_of_encodings = []
            for model in arguments.models:
                if helper_functions.get_mapping_name_to_class()[model].standard_encoding not in list_of_encodings:
                    list_of_encodings.append(helper_functions.get_mapping_name_to_class()[model].standard_encoding)
    return list_of_encodings


def get_list_of_encodings():
    """
    Get a list of all implemented encodings
    adapt if new encoding is added
    :return: List of all possible encodings
    """
    return ['raw', '012', 'onehot']


def get_base_encoding(encoding: str):
    """
    Function that checks which base encoding is needed to create required encoding
    adapt if new encoding is added
    :param encoding: required encoding
    :return: base encoding
    """
    if encoding in ('raw', '012', 'onehot'):
        return 'raw'
    else:  # adapt if new encoding is added
        raise Exception('No valid encoding. Can not determine base encoding')


def check_encoding_of_genotype(X: np.array):
    """
    Function to check the encoding of the genotype matrix
    :param X: genotype matrix
    :return: name of encoding
    """
    unique = np.unique(X)
    if all(z in ['A', 'C', 'G', 'T'] for z in unique.astype(str)):  # TODO heterozygous
        return 'raw'
    elif all(z in [0, 1, 2] for z in unique):
        return '012'


def encode_genotype(X: np.array, required_encoding: str, base_encoding: str):
    """
    compute required encoding of genotype matrix
    :param X: genotype matrix
    :param required_encoding: encoding of genotype matrix to create
    :param base_encoding: current encoding of X, needed if new encoding can be created in several ways
    :return: X in new encoding
    """
    if required_encoding == '012':
        return get_additive_encoding(X)
    elif required_encoding == 'onehot':
        return get_onehot_encoding(X)
    else:
        raise Exception('Only able to create additive or one-hot encoding.')


def get_additive_encoding(X: np.array):
    """
    generate genotype matrix in additive encoding:
    0: homozygous major allele,
    1: heterozygous
    2: homozygous minor allele
    :param X: genotype matrix in raw encoding, i.e. containing the alleles
    :return: X_012
    """
    alleles = []
    index_arr = []
    for j, col in enumerate(np.transpose(X)):
        unique, inv, counts = np.unique(col, return_counts=True, return_inverse=True)
        unique = unique.astype(str)
        boolean = (unique == 'A') | (unique == 'T') | (unique == 'C') | (unique == 'G')
        tmp = np.zeros(3)
        if len(unique) > 3:
            raise Exception('More than two alleles encountered at snp ' + str(j))
        elif len(unique) == 3:
            pairs = [['A', 'C'], ['A', 'G'], ['A', 'T'], ['C', 'G'], ['C', 'T'], ['G', 'T']]
            heterozygous_nuc = ['M', 'R', 'W', 'S', 'Y', 'K']
            hetero = unique[~boolean][0]
            homozygous = unique[boolean]
            for j, pair in enumerate(pairs):
                if all(h in pair for h in homozygous) and hetero != heterozygous_nuc[j]:
                    raise Exception('More than two alleles encountered at snp ' + str(j))
            tmp[~boolean] = 1.0
        tmp[np.argmin(counts[boolean])] = 2.0
        alleles.append(tmp)
        index_arr.append(inv)
    alleles = np.transpose(np.array(alleles))
    ind_arr = np.transpose(np.array(index_arr))
    cols = np.arange(alleles.shape[1])
    return alleles[ind_arr, cols]


def get_onehot_encoding(X: np.array):
    """
    Generate genotype matrix in one-hot encoding. If genotype matrix is homozygous, create 3d torch tensor with
    (samples, SNPs, 4), with 4 as the one-hot encoding
    A : [1,0,0,0]
    C : [0,1,0,0]
    G : [0,0,1,0]
    T : [0,0,0,1]
    If genotype matrix is heterozygous, create 3d torch tensor with (samples, SNPs, 10), with 10 as the one-hot encoding
    A : [1,0,0,0,0,0,0,0,0,0]
    C : [0,1,0,0,0,0,0,0,0,0]
    G : [0,0,1,0,0,0,0,0,0,0]
    K : [0,0,0,1,0,0,0,0,0,0]
    M : [0,0,0,0,1,0,0,0,0,0]
    R : [0,0,0,0,0,1,0,0,0,0]
    S : [0,0,0,0,0,0,1,0,0,0]
    T : [0,0,0,0,0,0,0,1,0,0]
    W : [0,0,0,0,0,0,0,0,1,0]
    Y : [0,0,0,0,0,0,0,0,0,1]
    :param X: genotype matrix in raw encoding, i.e. containing the alleles
    :return: X_onehot
    """
    unique, inverse = np.unique(X, return_inverse=True)
    inverse = inverse.reshape(X.shape)
    X_onehot = one_hot(torch.from_numpy(inverse)).numpy()
    return X_onehot
