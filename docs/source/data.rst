# easyPheno

[![Python 3.8](https://img.shields.io/badge/Python-3.8-3776AB)](https://www.python.org/downloads/release/python-388/)


## Input data
The minimal requirement is to provide a genotype and a phenotype file. We provide test data in the folder `data/test`. The genotype matrix needs to be imputed.
This Framework is designed to work with several genotype file formats. In a first step the genotype data will be saved
in a unified .h5 file which simplifies further processing. One can provide the following file formats:

### .hdf5/.h5/.h5py
The file has to contain the following keys:

- X_raw: genotype matrix in raw nucleotides (i.e. 'A', 'C', 'G', 'T', 'M', 'R', 'W', 'S', 'Y', 'K')
        with samples as rows and markers as columns
- sample_ids: vector containing corresponding sample ids in the same order as the genotype matrix
- snp_ids: vector containing the identifiers of all SNPs in the same order as the genotype matrix


### .csv
The first column should be the sample ids. The column names should be the SNP identifiers. The values should be the
genotype matrix in raw encoding (i.e. 'A', 'C', 'G', 'T', 'M', 'R', 'W', 'S', 'Y', 'K').


### PLINK
To use PLINK data, a .map and .ped file with the same prefix need to be in the same folder.
To run the Framework with PLINK files, you can use PREFIX.map or PREFIX.ped as option for the genotype file.


### binary PLINK
To use binary PLINK data, a .bed, .bim and .fam file with the same prefix need to be in the same folder.
To run the Framework with binary PLINK files, you can use PREFIX.bed, PREFIX.bim or PREFIX.fam as option for the
genotype file.

### phenotype file
The Framework currently only accepts .csv, .pheno and .txt files for the phenotype. Here the first column should contain
the sample ids. The remaining columns should contain the phenotype values with the phenotype name as column name.
For .txt and .pheno files it is assumed that the values are separated by a single space.

## Preprocessing

For each genotype-phenotype combination a separate index file will be created. This file contains the sample indices to
quickly re-match the genotype and phenotype matrices as well as datasets with indices for different data splits and MAF
filters. This way the data splits are the same for all models. Additionally, the sample ids and minor allele frequencies
for all SNPs are stored to easily create new MAF filters and data splits and append to the index file.
When first creating the index file, some standard values for the data splits and MAF filters will be used additionally
to the values specified by the user. The index file has the following format:

    'matched_data': {
        'y': matched phenotypic values,
        'matched_sample_ids': sample ids of matched genotype/phenotype,
        'X_index': indices of genotype matrix to redo matching,
        'y_index': indices of phenotype vector to redo matching,
        'ma_frequency': minor allele frequency of each SNP in genotype file
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

