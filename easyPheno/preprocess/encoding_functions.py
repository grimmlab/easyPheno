import numpy as np
import torch
from torch.nn.functional import one_hot

from ..utils import helper_functions


def get_encoding(models, user_encoding: str) -> list:
    """
    Get a list of all required encodings.

    :param models: models to consider
    :param user_encoding: encoding specified by the user

    :return: list of encodings
    """
    if models == 'all':
        # do all encodings if all models are specified
        list_of_encodings = get_list_of_encodings()
    else:
        # use the specified encoding or encodings of specified models
        if user_encoding is not None:
            list_of_encodings = [user_encoding]
        else:
            list_of_encodings = []
            for model in models:
                if helper_functions.get_mapping_name_to_class()[model].standard_encoding not in list_of_encodings:
                    list_of_encodings.append(helper_functions.get_mapping_name_to_class()[model].standard_encoding)
    return list_of_encodings


def get_list_of_encodings() -> list:
    """
    Get a list of all implemented encodings.

    ! Adapt if new encoding is added !

    :return: List of all possible encodings
    """
    return ['raw', '012', 'onehot', '101']


def get_base_encoding(encoding: str) -> str:
    """
    Check which base encoding is needed to create required encoding.

    ! Adapt if new encoding is added !

    :param encoding: required encoding

    :return: base encoding
    """
    if encoding in ('raw', '012', 'onehot', '101'):
        return 'raw'
    else:
        raise Exception('No valid encoding. Can not determine base encoding')


def check_encoding_of_genotype(X: np.array) -> str:
    """
    Check the encoding of the genotype matrix

    ! Adapt if new encoding is added !

    :param X: genotype matrix

    :return: encoding of the genotype matrix
    """
    unique = np.unique(X)
    if all(z in ['A', 'C', 'G', 'T', 'M', 'R', 'W', 'S', 'Y', 'K'] for z in unique.astype(str)):
        return 'raw'
    elif all(z in [0, 1, 2] for z in unique):
        return '012'


def encode_genotype(X: np.array, required_encoding: str) -> np.array:
    """
    Compute the required encoding of the genotype matrix

    ! Adapt if new encoding is added !

    :param X: genotype matrix
    :param required_encoding: encoding of genotype matrix to create

    :return: X in new encoding
    """
    if required_encoding == '012':
        return get_additive_encoding(X)
    elif required_encoding == 'onehot':
        return get_onehot_encoding(X)
    elif required_encoding == '101':
        return get_additive_encoding(X, '101')
    else:
        raise Exception('Only able to create additive or onehot encoding.')


def get_additive_encoding(X: np.array, style: str = '012') -> np.array:
    """
    Generate genotype matrix in additive encoding:

    - 0: homozygous major allele,
    - 1: heterozygous
    - 2: homozygous minor allele

    :param X: genotype matrix in raw encoding, i.e. containing the alleles

    :return: genotype matrix in additive encoding (X_012)
    """
    if style == '012':
        shift = 0
    elif style == '101':
        shift = 1
    else:
        raise Exception('only allow 012 or 101 as style for additive encoding')
    alleles = []
    index_arr = []
    pairs = [['A', 'C'], ['A', 'G'], ['A', 'T'], ['C', 'G'], ['C', 'T'], ['G', 'T']]
    heterozygous_nuc = ['M', 'R', 'W', 'S', 'Y', 'K']
    for j, col in enumerate(np.transpose(X)):
        unique, inv, counts = np.unique(col, return_counts=True, return_inverse=True)
        unique = unique.astype(str)
        boolean = (unique == 'A') | (unique == 'T') | (unique == 'C') | (unique == 'G')
        tmp = np.zeros(3) + shift
        if len(unique) > 3:
            raise Exception('More than two alleles encountered at snp ' + str(j))
        elif len(unique) == 3:
            hetero = unique[~boolean][0]
            homozygous = unique[boolean]
            for i, pair in enumerate(pairs):
                if all(h in pair for h in homozygous) and hetero != heterozygous_nuc[i]:
                    raise Exception('More than two alleles encountered at snp ' + str(i))
            tmp[~boolean] = 1.0 - shift
            tmp[np.argmin(counts[boolean])] = 2.0 - 3*shift
        elif len(unique) == 2:
            if list(unique) in pairs:
                tmp[np.argmin(counts)] = 2.0 - 3*shift
            else:
                tmp[(~boolean).nonzero()] = 1.0 - shift
        else:
            if unique[0] in heterozygous_nuc:
                tmp[0] = 1.0 - shift
        alleles.append(tmp)
        index_arr.append(inv)
    alleles = np.transpose(np.array(alleles))
    ind_arr = np.transpose(np.array(index_arr))
    cols = np.arange(alleles.shape[1])
    return alleles[ind_arr, cols]


def get_onehot_encoding(X: np.array) -> np.array:
    """
    Generate genotype matrix in onehot encoding. If genotype matrix is homozygous, create 3d torch tensor with
    (samples, SNPs, 4), with 4 as the onehot encoding

    - A : [1,0,0,0]
    - C : [0,1,0,0]
    - G : [0,0,1,0]
    - T : [0,0,0,1]

    If genotype matrix is heterozygous, create 3d torch tensor with (samples, SNPs, 10), with 10 as the onehot encoding

    - A : [1,0,0,0,0,0,0,0,0,0]
    - C : [0,1,0,0,0,0,0,0,0,0]
    - G : [0,0,1,0,0,0,0,0,0,0]
    - K : [0,0,0,1,0,0,0,0,0,0]
    - M : [0,0,0,0,1,0,0,0,0,0]
    - R : [0,0,0,0,0,1,0,0,0,0]
    - S : [0,0,0,0,0,0,1,0,0,0]
    - T : [0,0,0,0,0,0,0,1,0,0]
    - W : [0,0,0,0,0,0,0,0,1,0]
    - Y : [0,0,0,0,0,0,0,0,0,1]

    :param X: genotype matrix in raw encoding, i.e. containing the alleles

    :return: genotype matrix in onehot encoding (X_onehot)
    """
    unique, inverse = np.unique(X, return_inverse=True)
    inverse = inverse.reshape(X.shape)
    X_onehot = one_hot(torch.from_numpy(inverse)).numpy()
    return X_onehot
