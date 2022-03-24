import argparse
import datetime
import warnings
from optuna.exceptions import ExperimentalWarning

import optimization.optuna_optim
import preprocess.base_dataset
from utils import check_functions, print_functions, helper_functions
from preprocess import encoding_functions
import pprint
from preprocess import raw_data_functions


def run_pipeline(data_dir: str, genotype_matrix: str, phenotype_matrix: str, phenotype: str,
                 encoding: str = None, maf_percentage: int = 0, save_dir: str = None,
                 datasplit: str = 'nested-cv', n_outerfolds: int = 5, n_innerfolds: int = 5,
                 test_set_size_percentage: int = 20, val_set_size_percentage: int = 20,
                 models: list = None, n_trials: int = 100, save_final_model: bool = False,
                 batch_size: int = 32, n_epochs: int = None):
    """
    Run the whole optimization pipeline
    :param data_dir: data directory where the phenotype and genotype matrix are stored
    :param genotype_matrix: name of the genotype matrix including datatype ending
    :param phenotype_matrix: name of the phenotype matrix including datatype ending
    :param phenotype: name of the phenotype to predict
    :param encoding: encoding to use. Default is None, so standard encoding of each model will be used
    :param maf_percentage: threshold for MAF filter as percentage value. Default is 0, so no MAF filtering
    :param save_dir: directory for saving the results. Default is None, so same directory as data_dir
    :param datasplit: datasplit to use. Options are: nested-cv, cv-test, train-val-test
    :param n_outerfolds: number of outerfolds relevant for nested-cv
    :param n_innerfolds: number of folds relevant for nested-cv and cv-test
    :param test_set_size_percentage: size of the test set relevant for cv-test and train-val-test
    :param val_set_size_percentage: size of the validation set relevant for train-val-test
    :param models: list of models that should be optimized
    :param n_trials: number of trials for optuna
    :param save_final_model: specify if the final model should be saved
    :param batch_size: batch size for neural network models
    :param n_epochs: number of epochs for neural network models
    """
    if models is None:
        models = ['xgboost']
    # set save directory
    save_dir = data_dir if save_dir is None else save_dir
    if type(models) == list and models[0] == 'all':
        models = 'all'
    if type(models) != list and models != 'all':
        models = [models]

    # Checks and Raw Data Input Preparation #
    # Check all arguments
    check_functions.check_all_specified_arguments(arguments=locals())
    # prepare all data files
    raw_data_functions.prepare_data_files(
        data_dir=data_dir, genotype_matrix_name=genotype_matrix, phenotype_matrix_name=phenotype_matrix,
        phenotype=phenotype, datasplit=datasplit, n_outerfolds=n_outerfolds, n_innerfolds=n_innerfolds,
        test_set_size_percentage=test_set_size_percentage, val_set_size_percentage=val_set_size_percentage,
        models=models, user_encoding=encoding, maf_percentage=maf_percentage
    )

    # Optimization Pipeline #
    helper_functions.set_all_seeds()
    models_to_optimize = helper_functions.get_list_of_implemented_models() if models == 'all' else models
    if len(models_to_optimize) > 1:
        models_to_optimize = helper_functions.sort_models_by_encoding(models_list=models_to_optimize)
    model_overview = {}
    start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for optim_run, current_model_name in enumerate(models_to_optimize):
        encoding = encoding if encoding is not None \
            else helper_functions.get_mapping_name_to_class()[current_model_name].standard_encoding
        if optim_run == 0:
            print('----- Starting dataset preparation -----')
            dataset = preprocess.base_dataset.Dataset(
                data_dir=data_dir, genotype_matrix_name=genotype_matrix, phenotype_matrix_name=phenotype_matrix,
                phenotype=phenotype, datasplit=datasplit, n_outerfolds=n_outerfolds, n_innerfolds=n_innerfolds,
                test_set_size_percentage=test_set_size_percentage, val_set_size_percentage=val_set_size_percentage,
                encoding=encoding, maf_percentage=maf_percentage
            )
            task = 'classification' if helper_functions.test_likely_categorical(dataset.y_full) else 'regression'
            print_functions.print_config_info(arguments=locals(), dataset=dataset, task=task)
        else:
            if dataset.encoding != encoding:
                print('----- Load new dataset encoding -----')
                dataset = preprocess.base_dataset.Dataset(
                    data_dir=data_dir, genotype_matrix_name=genotype_matrix, phenotype_matrix_name=phenotype_matrix,
                    phenotype=phenotype, datasplit=datasplit, n_outerfolds=n_outerfolds, n_innerfolds=n_innerfolds,
                    test_set_size_percentage=test_set_size_percentage, val_set_size_percentage=val_set_size_percentage,
                    encoding=encoding, maf_percentage=maf_percentage
                )
        optuna_run = optimization.optuna_optim.OptunaOptim(
            save_dir=save_dir, genotype_matrix_name=genotype_matrix, phenotype_matrix_name=phenotype_matrix,
            phenotype=phenotype, n_outerfolds=n_outerfolds, n_innerfolds=n_innerfolds,
            val_set_size_percentage=val_set_size_percentage, test_set_size_percentage=test_set_size_percentage,
            maf_percentage=maf_percentage, n_trials=n_trials, save_final_model=save_final_model, batch_size=batch_size,
            n_epochs=n_epochs, task=task, start_time=start_time, current_model_name=current_model_name, dataset=dataset)
        print('### Starting Optuna Optimization for ' + current_model_name + ' ###')
        overall_results = optuna_run.run_optuna_optimization()
        print('### Finished Optuna Optimization for ' + current_model_name + ' ###')
        model_overview[current_model_name] = overall_results
    print('# Optimization runs done for models ' + str(models_to_optimize))
    print('Results overview on the test set(s)')
    pprint.PrettyPrinter(depth=4).pprint(model_overview)
    path_overview_file = \
        optuna_run.base_path[:[index for index, letter in enumerate(optuna_run.base_path) if letter == '/'][-2]] + \
        '/Results_overiew_' + '_'.join(models) + '.csv'
    helper_functions.save_model_overview_dict(model_overview=model_overview, save_path=path_overview_file)


if __name__ == '__main__':
    """
    Run file to start the whole procedure:
            Parameter Plausibility Check
            Check and prepare data files
            Bayesian optimization for each chosen model
    """
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=ExperimentalWarning)
    # User Input #
    parser = argparse.ArgumentParser()
    # Input Params #
    parser.add_argument("-dd", "--data_dir", type=str,
                        default='data/test',
                        help="Provide the full path of your data directory "
                             "(that contains the geno- and phenotype files).")
    parser.add_argument("-sd", "--save_dir", type=str, default='/home/fhaselbeck',
                        help="Provide the full path of the directory in which you want to save your results. "
                             "Default is same as data_dir")
    parser.add_argument("-gm", "--genotype_matrix", type=str, default='x_matrix.h5',
                        help="specify the name (including data type suffix) of the genotype matrix to be used. "
                             "Needs to be located in the subfolder data/ of the specified base directory" +
                             "For more info regarding the required format see our documentation at GitHub")
    parser.add_argument("-pm", "--phenotype_matrix", type=str, default='y_matrix.csv',
                        help="specify the name (including data type suffix) of the phenotype matrix to be used. "
                             "Needs to be located in the subfolder data/ of the specified base directory" +
                             "For more info regarding the required format see our documentation at GitHub")
    parser.add_argument("-ph", "--phenotype", nargs='+', type=str, default=['continuous_values'],
                        help="specify the name of the phenotype to be predicted. "
                             "Multiple phenotypes can also be chosesn if they are in the same phenotype matrix")
    parser.add_argument("-enc", "--encoding", type=str, default=None,
                        help="specify the encoding to use. Caution: has to be a possible encoding for the model to use."
                             "Valid arguments are: " + str(encoding_functions.get_list_of_encodings()))

    # Preprocess Params #
    parser.add_argument("-maf", "--maf_percentage", type=int, default=10,
                        help="specify the minor allele frequency (as percentage value). "
                             "specify 0 if you do not want a maf filter.")
    parser.add_argument("-split", "--datasplit", type=str, default='nested-cv',
                        help="specify the data slit to use: 'nested-cv' | 'cv-test' | 'train-val-test'"
                             "Default values are 5 folds, train-test-split to 80/20 and train-val-test to 60/20/20")
    parser.add_argument("-testperc", "--test_set_size_percentage", type=int, default=20,
                        help="specify the size of the test set in percentage. "
                             "Standard is 20, only relevant for 'cv-test' and 'train-val-test'")
    parser.add_argument("-valperc", "--val_set_size_percentage", type=int, default=20,
                        help="specify the size of the validation set in percentage. "
                             "Standard is 20, only relevant 'train-val-test'")
    parser.add_argument("-of", "--n_outerfolds", type=int, default=5,
                        help="specify the number of outerfolds to use for 'nested_cv'"
                             "Standard is 20, only relevant 'nested_cv'")
    parser.add_argument("-folds", "--n_innerfolds", type=int, default=5,
                        help="specify the number of innerfolds/folds to use for 'nested_cv' respectively 'cv-test'"
                             "Standard is 5, only relevant 'nested_cv' and 'cv-test'")

    # Model and Optimization Params #
    parser.add_argument("-mod", "--models", nargs='+', type=str, default=['xgboost'],
                        help="specify the models to optimize: all or naming according to source file name. "
                             "Multiple models can be selected by just naming multiple model names, "
                             "e.g. --models mlp xgboost. "
                             "The following are available: " + str(helper_functions.get_list_of_implemented_models()))
    parser.add_argument("-tr", "--n_trials", type=int, default=100,
                        help="number of trials for optuna")
    parser.add_argument("-sf", "--save_final_model", type=bool, default=False,
                        help="save the final model to hard drive "
                             "(caution: some models may use a lot of disk space, "
                             "unfitted models that can be retrained are already saved by default)")

    # Only relevant for Neural Networks #
    parser.add_argument("-bs", "--batch_size", type=int, default=32,
                        help="Only relevant for neural networks: define the batch size. If nothing is specified,"
                             "it will be considered as a hyperparameter for optimization")
    parser.add_argument("-ep", "--n_epochs", type=int, default=None,
                        help="Only relevant for neural networks: define the number of epochs. If nothing is specified,"
                             "it will be considered as a hyperparameter for optimization")
    args = vars(parser.parse_args())
    phenotypes = args["phenotype"]

    for phenotype in phenotypes:
        args["phenotype"] = phenotype
        try:
            run_pipeline(**args)
        except Exception as exc:
            print("Failure when running pipeline for " + phenotype)
            print(exc)
            continue
