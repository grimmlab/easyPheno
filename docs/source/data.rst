Data Guide
===================
To run easyPheno on your data, you need to provide a fully-imputed genotype file and a corresponding phenotype file both stored in the same data directory.
easyPheno is designed to work with several genotype and phenotype file types.

Genotype files
----------------
Independent of the original file type, when loading it the first time, the genotype data will be saved to the data directory
in a unified H5 file with the same prefix as the original genotype file to simplify further processing and future runs.
easyPheno accepts the following genotype file types:

HDF5 / H5 / H5PY
~~~~~~~~~~~~~~~~~~~
The file has to contain the following keys:

- **X_raw:** genotype matrix in IUPAC nucleotide code (i.e. 'A', 'C', 'G', 'T', 'M', 'R', 'W', 'S', 'Y', 'K') with *samples* as *rows* and *markers* as *columns*
- **sample_ids:** vector containing corresponding sample ids in the same order as the rows of the genotype matrix
- **snp_ids:** vector containing the identifiers of all SNPs in the same order as the columns of the genotype matrix

CSV
~~~~~
The *first column* must contain the unique **sample id** for each sample. The *column names* should be the **SNP identifiers**.
The *values* should be the **genotype matrix** in IUPAC nucleotide code (i.e. 'A', 'C', 'G', 'T', 'M', 'R', 'W', 'S', 'Y', 'K'),
with *samples* as *rows* and *markers* as *columns*

PLINK
~~~~~~~
To use PLINK data, a **.map** and **.ped** file with the same prefix need to be in the data directory.
To run the framework with PLINK files, you can use PREFIX.map or PREFIX.ped as option for the genotype file.
(See `PLINK <https://www.cog-genomics.org/plink/>`_ for more info on the file type)

binary PLINK
~~~~~~~~~~~~~~
To use binary PLINK data, a **.bed**, **.bim** and **.fam** file with the same prefix need to be in the data directory.
To run the framework with binary PLINK files, you can use PREFIX.bed, PREFIX.bim or PREFIX.fam as option for the
genotype file. (See `PLINK <https://www.cog-genomics.org/plink/>`_ for more info on the file type)

Phenotype file
---------------
easyPheno currently only accepts **.csv**, **.pheno** and **.txt** files for the phenotype. For .txt and .pheno files it
is assumed that the values are separated by a single space.
A phenotype file can contain several phenotypes.
The *first column* must always contain the **sample ids** corresponding to the genotype matrix (need not be in the same order).
The remaining columns should contain the **phenotype values** with the **phenotype name** as *column name*.

Preprocessing
----------------
For each genotype-phenotype combination a separate index file will be created. This file contains the sample indices to
quickly re-match the genotype and phenotype matrices as well as datasets with indices for different data splits and
minor-allele-frequency filters. This way the data splits are the same for all models. Additionally, the sample ids and
minor allele frequencies for all SNPs are stored to easily create new MAF filters and data splits and append to the index file.
To test the model on new unseen data, the index file also contains the final SNP ids used by each model, sorted by used
encoding and minor-allele-frequency.
When first creating the index file, some standard values for the data splits and MAF filters will be used additionally
to the values specified by the user. The index file has the following format:

    .. code-block::

        'matched_data': {
            'y': matched phenotypic values,
            'matched_sample_ids': sample ids of matched genotype/phenotype,
            'X_index': indices of genotype matrix to redo matching,
            'y_index': indices of phenotype vector to redo matching,
            'ma_frequency': minor allele frequency of each SNP in genotype file
            'final_snp_ids':{
                '{encoding}':{
                    'maf_{maf_percentage}_snp_ids'
                    }
                }
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

