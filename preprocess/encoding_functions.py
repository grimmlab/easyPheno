import numpy as np

def encode_raw_genotype(X: np.array, encoding: str):
    """

    :param X:
    :param encoding:
    :return:
    """
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
    :param X: genotype matrix in raw encoding, i.e. containing the alleles
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
    :param X: genotype matrix in raw encoding, i.e. containing the alleles
    :return: X_onehot
    """
    # TODO
    raise NotImplementedError