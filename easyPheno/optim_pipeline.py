import datetime
import pprint
import os

from easyPheno.utils import check_functions, print_functions, helper_functions
from easyPheno.preprocess import raw_data_functions, base_dataset
from easyPheno.optimization import optuna_optim, paramfree_fitting
from easyPheno.model import _param_free_base_model


def run(data_dir: str, genotype_matrix: str, phenotype_matrix: str, phenotype: str,
        encoding: str = None, maf_percentage: int = 0, save_dir: str = None,
        datasplit: str = 'nested-cv', n_outerfolds: int = 5, n_innerfolds: int = 5,
        test_set_size_percentage: int = 20, val_set_size_percentage: int = 20,
        models: list = None, n_trials: int = 100, save_final_model: bool = False,
        batch_size: int = 32, n_epochs: int = 100000):
    """
    Run the whole optimization pipeline

    :param data_dir: data directory where the phenotype and genotype matrix are stored
    :param genotype_matrix: name of the genotype matrix including datatype ending
    :param phenotype_matrix: name of the phenotype matrix including datatype ending
    :param phenotype: name of the phenotype to predict
    :param encoding: encoding to use. Default is None, so standard encoding of each model will be used. Options are: '012', 'onehot', 'raw'
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
    save_dir = os.getcwd() + '/' + save_dir if (len(save_dir) == 0 or save_dir[0] != '/') else save_dir
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
    models_start_time = '+'.join(models_to_optimize) + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    user_encoding = encoding
    for optim_run, current_model_name in enumerate(models_to_optimize):
        encoding = user_encoding if user_encoding is not None \
            else helper_functions.get_mapping_name_to_class()[current_model_name].standard_encoding
        if optim_run == 0:
            print('----- Starting dataset preparation -----')
            dataset = base_dataset.Dataset(
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
                dataset = base_dataset.Dataset(
                    data_dir=data_dir, genotype_matrix_name=genotype_matrix, phenotype_matrix_name=phenotype_matrix,
                    phenotype=phenotype, datasplit=datasplit, n_outerfolds=n_outerfolds, n_innerfolds=n_innerfolds,
                    test_set_size_percentage=test_set_size_percentage, val_set_size_percentage=val_set_size_percentage,
                    encoding=encoding, maf_percentage=maf_percentage
                )
        if issubclass(helper_functions.get_mapping_name_to_class()[current_model_name],
                      _param_free_base_model.ParamFreeBaseModel):
            print('### Starting Model Fitting for ' + current_model_name + ' ###')
            optim_run = paramfree_fitting.ParamFreeFitting(
                save_dir=save_dir, genotype_matrix_name=genotype_matrix, phenotype_matrix_name=phenotype_matrix,
                phenotype=phenotype, n_outerfolds=n_outerfolds, n_innerfolds=n_innerfolds,
                val_set_size_percentage=val_set_size_percentage, test_set_size_percentage=test_set_size_percentage,
                maf_percentage=maf_percentage, save_final_model=save_final_model,
                task=task, models_start_time=models_start_time, current_model_name=current_model_name, dataset=dataset
            )
            overall_results = optim_run.run_fitting()
            print('### Finished Model Fitting for ' + current_model_name + ' ###')
        else:
            optim_run = optuna_optim.OptunaOptim(
                save_dir=save_dir, genotype_matrix_name=genotype_matrix, phenotype_matrix_name=phenotype_matrix,
                phenotype=phenotype, n_outerfolds=n_outerfolds, n_innerfolds=n_innerfolds,
                val_set_size_percentage=val_set_size_percentage, test_set_size_percentage=test_set_size_percentage,
                maf_percentage=maf_percentage, n_trials=n_trials, save_final_model=save_final_model, batch_size=batch_size,
                n_epochs=n_epochs, task=task, models_start_time=models_start_time, current_model_name=current_model_name,
                dataset=dataset)
            print('### Starting Optuna Optimization for ' + current_model_name + ' ###')
            overall_results = optim_run.run_optuna_optimization()
            print('### Finished Optuna Optimization for ' + current_model_name + ' ###')
        model_overview[current_model_name] = overall_results
    print('# Optimization runs done for models ' + str(models_to_optimize))
    print('Results overview on the test set(s)')
    pprint.PrettyPrinter(depth=4).pprint(model_overview)
    path_overview_file = \
        optim_run.base_path[:[index for index, letter in enumerate(optim_run.base_path) if letter == '/'][-2]] + \
        '/Results_overiew_' + '_'.join(models) + '.csv'
    helper_functions.save_model_overview_dict(model_overview=model_overview, save_path=path_overview_file)
