import argparse
import datetime

import optimization.optuna_optim
import preprocess.base_dataset
from utils import check_functions, print_functions, helper_functions
from preprocess import encoding_functions
import pprint
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
    parser.add_argument("-data_dir", "--data_dir", type=str, default='/bit_storage/Workspace/Maura/PhenotypePred/data',
                        help="Provide the full path of your data directory "
                             "(that contains the geno- and phenotype files).")
    parser.add_argument("-save_dir", "--save_dir", type=str, default='/home/fhaselbeck/Work/phenotypepred/',
                        help="Provide the full path of the directory in which you want to save your results. "
                             "Default is same as data_dir")
    parser.add_argument("-geno_matrix", "--genotype_matrix", type=str, default='x_matrix_big.h5',
                        help="specify the name (including data type suffix) of the genotype matrix to be used. "
                             "Needs to be located in the subfolder data/ of the specified base directory" +
                             "For more info regarding the required format see our documentation at GitHub")
    parser.add_argument("-pheno_matrix", "--phenotype_matrix", type=str, default='study_12_values.csv',
                        help="specify the name (including data type suffix) of the phenotype matrix to be used. "
                             "Needs to be located in the subfolder data/ of the specified base directory" +
                             "For more info regarding the required format see our documentation at GitHub")
    parser.add_argument("-phenotype", "--phenotype", type=str, default='Fake',
                        help="specify the name of the phenotype to be predicted")
    parser.add_argument("-enc", "--encoding", type=str, default=None,
                        help="specify the encoding to use. Caution: has to be a possible encoding for the model to use."
                             "Valid arguments are: " + str(encoding_functions.get_list_of_encodings()))

    # Preprocess Params #
    parser.add_argument("-maf", "--maf_percentage", type=int, default=1,
                        help="specify the minor allele frequency (as percentage value). "
                             "specify 0 if you do not want a maf filter.")
    parser.add_argument("-datasplit", "--datasplit", type=str, default='cv-test',
                        help="specify the data slit to use: 'nested-cv' | 'cv-test' | 'train-val-test'"
                             "Default values are 5 folds, train-test-split to 80/20 and train-val-test to 60/20/20")
    parser.add_argument("-testperc", "--test_set_size_percentage", type=int, default=20,
                        help="specify the size of the test set in percentage. "
                             "Standard is 20, only relevant for 'cv-test' and 'train-val-test'")
    parser.add_argument("-valperc", "--validation_set_size_percentage", type=int, default=20,
                        help="specify the size of the validation set in percentage. "
                             "Standard is 20, only relevant 'train-val-test'")
    parser.add_argument("-outerfolds", "--n_outerfolds", type=int, default=3,
                        help="specify the number of outerfolds to use for 'nested_cv'"
                             "Standard is 20, only relevant 'nested_cv'")
    parser.add_argument("-folds", "--n_innerfolds", type=int, default=3,
                        help="specify the number of innerfolds/folds to use for 'nested_cv' respectively 'cv-test'"
                             "Standard is 5, only relevant 'nested_cv' and 'cv-test'")

    # Model and Optimization Params #
    parser.add_argument("-models", "--models", nargs='+', type=str, default=['xgboost'],
                        help="specify the models to optimize: all or naming according to source file name. "
                             "Multiple models can be selected by just naming multiple model names, "
                             "e.g. --models mlp xgboost. "
                             "The following are available: " + str(helper_functions.get_list_of_implemented_models()))
    parser.add_argument("-trials", "--n_trials", type=int, default=10,
                        help="number of trials for optuna")
    parser.add_argument("-save_final_model", "--save_final_model", type=bool, default=False,
                        help="save the final model to hard drive "
                             "(caution: some models may use a lot of disk space, "
                             "unfitted models that can be retrained are already saved by default)")

    # Only relevant for Neural Networks #
    parser.add_argument("-batch_size", "--batch_size", type=int, default=None,
                        help="Only relevant for neural networks: define the batch size. If nothing is specified,"
                             "it will be considered as a hyperparameter for optimization")
    parser.add_argument("-n_epochs", "--n_epochs", type=int, default=None,
                        help="Only relevant for neural networks: define the number of epochs. If nothing is specified,"
                             "it will be considered as a hyperparameter for optimization")
    args = parser.parse_args()

    # set save directory
    args.save_dir = args.data_dir if args.save_dir is None else args.save_dir
    if args.models[0] == 'all':
        args.models = 'all'
    ### Checks and Raw Data Input Preparation ###
    # Check all arguments
    check_functions.check_all_specified_arguments(arguments=args)
    # prepare all data files
    raw_data_functions.prepare_data_files(arguments=args)

    ### Optimization Pipeline ###
    helper_functions.set_all_seeds()
    models_to_optimize = helper_functions.get_list_of_implemented_models() if args.models == 'all' else args.models
    model_overview = {}
    start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for optim_run, current_model_name in enumerate(models_to_optimize):
        encoding = args.encoding if args.encoding is not None \
            else helper_functions.get_mapping_name_to_class()[current_model_name].standard_encoding
        dataset = preprocess.base_dataset.Dataset(arguments=args, encoding=encoding)
        task = 'classification' if helper_functions.test_likely_categorical(dataset.y_full) else 'regression'
        if optim_run == 0:
            print_functions.print_config_info(arguments=args, dataset=dataset, task=task)
        optuna_run = optimization.optuna_optim.OptunaOptim(arguments=args, task=task, start_time=start_time,
                                                           current_model_name=current_model_name, dataset=dataset)
        print('### Starting Optuna Optimization for ' + current_model_name + ' ###')
        overall_results = optuna_run.run_optuna_optimization()
        print('### Finished Optuna Optimization for ' + current_model_name + ' ###')
        model_overview[current_model_name] = overall_results
    print('# Optimization runs done for models ' + str(models_to_optimize))
    print('Results overview on the test set(s)')
    pprint.PrettyPrinter(depth=4).pprint(model_overview)
    path_overview_file = \
        optuna_run.base_path[:[index for index, letter in enumerate(optuna_run.base_path) if letter == '/'][-2]] + \
        '/Results_overiew_' + '_'.join(args.models) + '.csv'
    helper_functions.save_model_overview_dict(model_overview=model_overview, save_path=path_overview_file)
