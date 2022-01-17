import argparse
from utils import check_functions, print_functions, helper_functions
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

    ### User Input ###
    parser = argparse.ArgumentParser()
    # Input Params #
    parser.add_argument("-base_dir", "--base_dir", type=str, default='/bit_storage/Workspace/Maura/PhenotypePred/',
                        help="Provide the full path of your base directory (parent directory of the data folder that"
                             "contains your genotype and phenotype data). "
                             "Results will be saved in subdirectories starting there.")
    parser.add_argument("-geno_matrix", "--genotype_matrix", type=str, default='FT10_small.h5',
                        help="specify the name (including data type suffix) of the genotype matrix to be used. "
                             "Needs to be located in the subfolder data/ of the specified base directory" +
                             "For more info regarding the required format see our documentation at GitHub")
    parser.add_argument("-pheno_matrix", "--phenotype_matrix", type=str, default='study_12_values.csv',
                        help="specify the name (including data type suffix) of the phenotype matrix to be used. "
                             "Needs to be located in the subfolder data/ of the specified base directory" +
                             "For more info regarding the required format see our documentation at GitHub")
    parser.add_argument("-phenotype", "--phenotype", type=str, default='FT10',
                        help="specify the name of the phenotype to be predicted")

    # Preprocess Params #
    parser.add_argument("-maf", "--maf_percentage", type=int, default=1,
                        help="specify the minor allele frequency (as percentage value). "
                             "specify 0 if you do not want a maf filter.")
    # TODO: Als Standard werden 1, 3, 5 und 10 Prozent in den .h5 Dateien abgespeichert
    #  -> wenn ein neuer Wert von einem Nutzer dazu kommt, wird der in der .h5 nochmal hinzugefügt?
    parser.add_argument("-datasplit", "--datasplit", type=str, default='nested_cv',
                        help="specify the data slit to use: 'nested_cv' | 'cv-test' | 'train-val-test'"
                             "Default values are 5 folds, train-test-split to 80/20 and train-val-test to 60/20/20")
    parser.add_argument("-testperc", "--test_set_size_percentage", type=int, default=20,
                        help="specify the size of the test set in percentage. "
                             "Standard is 20, only relevant for 'cv-test' and 'train-val-test'")
    parser.add_argument("-valperc", "--validation_set_size_percentage", type=int, default=20,
                        help="specify the size of the validation set in percentage. "
                             "Standard is 20, only relevant 'train-val-test'")
    parser.add_argument("-outerfolds", "--n_outerfolds", type=int, default=5,
                        help="specify the number of outerfolds to use for 'nested_cv'"
                             "Standard is 20, only relevant 'nested_cv'")
    parser.add_argument("-folds", "--n_innerfolds", type=int, default=5,
                        help="specify the number of innerfolds/folds to use for 'nested_cv' respectively 'cv-test'"
                             "Standard is 5, only relevant 'nested_cv' and 'cv-test'")

    # Model and Optimization Params #
    parser.add_argument("-model", "--model", type=str, default='cnn',
                        help="specify the model(s) to optimize: 'all' or naming according to source file name "
                             "(without suffix .py) in subfolder model of this repo")
    parser.add_argument("-trials", "--n_trials", type=int, default=50,
                        help="number of trials for optuna")
    args = parser.parse_args()

    ### Checks and Raw Data Input Preparation ###
    # Check all arguments
    check_functions.check_all_specified_arguments(arguments=args)
    # Check and create subdirectories
    check_functions.check_and_create_directories(arguments=args)
    # prepare all data files
    raw_data_functions.prepare_data_files(arguments=args)
    # Print info for current config
    print_functions.print_config_info()

    ### Optimization Pipeline ###
    models_to_optimize = helper_functions.get_list_of_implemented_models() if args.model == 'all' else [args.model]
    for current_model in models_to_optimize:
        ...
        # Daten in richtigem encoding laden
        # Modell instantiieren
        # Optimierung laufen lassen
