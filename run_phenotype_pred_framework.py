import argparse
from utils import check_functions, print_functions
from preprocess import raw_data_functions

if __name__ == '__main__':
    """
    Run file to start the whole procedure:
        1. Parameter Plausibility Check
        2. Load Data
        3. preprocess
        4. Model Init
        5. Bayesian Optimization (Optuna)
        6. Evaluation
    """
    # Insert your base directory here
    base_dir = '/bit_storage/Workspace/Maura/PhenotypePred/'

    parser = argparse.ArgumentParser()
    ### Input Params ###
    parser.add_argument("-geno_matrix", "--genotype_matrix", type=str, default='FT10_small.h5',
                        help="specify the name of the genotype matrix to be used. "
                             "Needs to be located at " + base_dir + 'data/' +
                             "For more info regarding the required format see our documentation at GitHub")
    parser.add_argument("-pheno_matrix", "--phenotype_matrix", type=str, default='study_12_values.csv',
                        help="specify the name of the phenotype matrix to be used. "
                             "Needs to be located at " + base_dir + '/data/' +
                             "For more info regarding the required format see our documentation at GitHub")
    parser.add_argument("-phenotype", "--phenotype", type=str, default='FT10',
                        help="specify the name of the phenotype to be predicted")

    ### preprocess Params ###
    parser.add_argument("-maf", "--maf_percentage", type=int, default=1,
                        help="specify the minor allele frequency (as percentage value). "
                             "specify 0 if you do not want a maf filter.")
    # TODO: Als Standard werden 1, 3, 5 und 10 Prozent in den .h5 Dateien abgespeichert
    #  -> wenn ein neuer Wert von einem Nutzer dazu kommt, wird der in der .h5 nochmal hinzugefügt?
    parser.add_argument("-datasplit", "--datasplit", type=str, default='nested_cv',
                        help="specify the data slit to use: 'nested_cv' | 'cv-test' | 'train-val-test'"
                             "number of folds are fixed to 5, train-test-split to 80/20 and train-val-test to 60/20/20")

    ### Model and Optimization Params ###
    parser.add_argument("-model", "--model", type=str, default='cnn',
                        help="specify the model(s) to optimize: 'all' or naming according to source file name "
                             "(without suffix .py) in subfolder model of this repo")
    parser.add_argument("-trials", "--n_trials", type=int, default=50,
                        help="number of trials for optuna")

    args = parser.parse_args()

    # Check all arguments
    check_functions.check_all_specified_arguments(base_dir=base_dir, arguments=args)

    # Check and create subdirectories
    check_functions.check_and_create_directories(base_dir=base_dir, arguments=args)

    # Check and possibly transform genotype matrix format
    raw_data_functions.check_transform_format_genotype_matrix()

    # Match genotype and phenotype matrix
    raw_data_functions.genotype_phenotype_matching()

    # Print info for current config
    print_functions.print_config_info()



