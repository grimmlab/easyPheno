import numpy as np
import h5py
import pandas as pd
import pathlib
import datetime

from ..preprocess import base_dataset, raw_data_functions
from ..utils import helper_functions, check_functions
from ..model import _model_functions, _torch_model, _tensorflow_model, _base_model
from ..evaluation import eval_metrics
from . import feat_importance, results_analysis


def apply_final_model(results_directory_model: str, old_data_dir: str, new_data_dir: str,
                      new_genotype_matrix: str, new_phenotype_matrix: str = None, save_dir: str = None):
    """
    Apply a final model on a new dataset. It will be applied to the whole dataset.
    So the main purpose of this function is, if you get new samples you want to predict on.
    If the final model was saved, this will be used for inference on the new dataset.
    Otherwise, it will be retrained on the initial dataset and then used for inference on the new dataset.

    The new dataset will be filtered for the SNP ids that the model was initially trained on.

    CAUTION: the SNPs of the old and the new dataset have to be the same!

    :param results_directory_model: directory that contains the model results that you want to use
    :param old_data_dir: directory that contains the data that the model was trained on
    :param new_data_dir: directory that contains the new genotype and phenotype matrix
    :param new_genotype_matrix: new genotype matrix (incl. file suffix)
    :param new_phenotype_matrix: optional - new phenotype matrix (incl. file suffix) to directly get metrics for predictions.
    :param save_dir: directory to store the results
    """
    results_directory_model = pathlib.Path(results_directory_model)
    old_data_dir = pathlib.Path(old_data_dir)
    new_data_dir = pathlib.Path(new_data_dir)
    save_dir = pathlib.Path(save_dir)

    # Check user inputs
    print("Checking user inputs")
    if not check_functions.check_exist_directories(
            list_of_dirs=[results_directory_model, new_data_dir, save_dir, old_data_dir]):
        raise Exception("See output above. Problems with specified directories")

    result_folder_name = results_directory_model.parts[-3] if 'nested' in str(results_directory_model) \
        else results_directory_model.parts[-2]
    model_name = results_directory_model.parts[-1]
    if 'fromR' in model_name:
        import easypheno.model
        easypheno.model.__all__.extend(['_bayesfromR', 'bayesAfromR', 'bayesBfromR', 'bayesCfromR'])
    print("DEBUGGING")
    print(result_folder_name)
    datasplit, n_outerfolds, n_innerfolds, val_set_size_percentage, test_set_size_percentage, maf_perc = \
        helper_functions.get_datasplit_config_info_for_resultfolder(resultfolder=result_folder_name)
    nested_offset = -1 if 'nested' in datasplit else 0
    phenotype = results_directory_model.parts[-3 + nested_offset]
    old_genotype_matrix = results_directory_model.parts[-5 + nested_offset]
    old_phenotype_matrix = results_directory_model.parts[-4 + nested_offset]
    list_of_files = [] if new_phenotype_matrix is None else [new_data_dir.joinpath(new_phenotype_matrix)]
    list_of_files.extend([new_data_dir.joinpath(new_genotype_matrix), old_data_dir.joinpath(base_dataset.Dataset.get_index_file_name(genotype_matrix_name=old_genotype_matrix, phenotype_matrix_name=old_phenotype_matrix,phenotype=phenotype))])
    if not check_functions.check_exist_files(list_of_files=list_of_files):
        raise Exception("See output above. Problems with specified files.")

    # Prepare the new data
    print("Preparing the new dataset")
    raw_data_functions.prepare_data_files(
        data_dir=new_data_dir, genotype_matrix_name=new_genotype_matrix, phenotype_matrix_name=new_phenotype_matrix,
        phenotype=phenotype, datasplit=datasplit, n_outerfolds=n_outerfolds, n_innerfolds=n_innerfolds,
        test_set_size_percentage=test_set_size_percentage, val_set_size_percentage=val_set_size_percentage,
        models=[model_name], user_encoding=None, maf_percentage=maf_perc
    )
    encoding = helper_functions.get_mapping_name_to_class()[model_name].standard_encoding
    if new_phenotype_matrix is None:
        new_dataset = base_dataset.Datasetinfonly(data_dir=new_data_dir, genotype_matrix_name=new_genotype_matrix, encoding=encoding, maf_percentage=maf_perc, do_snp_filters=False)
    else:
        new_dataset = base_dataset.Dataset(
            data_dir=new_data_dir, genotype_matrix_name=new_genotype_matrix, phenotype_matrix_name=new_phenotype_matrix,
            phenotype=phenotype, datasplit=datasplit, n_outerfolds=n_outerfolds, n_innerfolds=n_innerfolds,
            test_set_size_percentage=test_set_size_percentage, val_set_size_percentage=val_set_size_percentage,
            encoding=encoding, maf_percentage=maf_perc, do_snp_filters=False
        )

    # Check and filter SNPids in comparison with old dataset
    with h5py.File(old_data_dir.joinpath(
            base_dataset.Dataset.get_index_file_name(genotype_matrix_name=old_genotype_matrix,
                                                     phenotype_matrix_name=old_phenotype_matrix,
                                                     phenotype=phenotype)), "r") as f:
        old_dataset_snp_ids = f[f'matched_data/final_snp_ids/{encoding}/maf_{maf_perc}_snp_ids'][:].astype(str)
    if not check_functions.compare_snp_id_vectors(snp_id_vector_big_equal=new_dataset.snp_ids,
                                                  snp_id_vector_small_equal=old_dataset_snp_ids):
        raise Exception('SNPids of initial dataset and new dataset do not match.')
    old_dataset_snp_ids = np.asarray(old_dataset_snp_ids, dtype=new_dataset.snp_ids.dtype).flatten()
    _, ids_to_keep = \
        (np.reshape(old_dataset_snp_ids, (old_dataset_snp_ids.shape[0], 1)) == new_dataset.snp_ids).nonzero()
    new_dataset.X_full = new_dataset.X_full[:, ids_to_keep]

    # Prepare the model
    outerfold_number = int(results_directory_model.parent.parts[-1].split('_')[1]) if 'nested' in datasplit else 0
    models = results_directory_model.parts[-2 + nested_offset].split('_')[3].split('+')
    results_file_path = \
        results_directory_model.parents[0 - nested_offset].joinpath('Results_overview_' + '_'.join(models) + '.csv')
    full_model_path = results_directory_model.joinpath("final_retrained_model")
    print("DEBUGGING")
    print(full_model_path)
    if full_model_path.is_file():
        print("Loading saved model")
        model = _model_functions.load_model(path=results_directory_model, filename=full_model_path.parts[-1])
    else:
        print("Retraining model")
        print("Loading old dataset")
        old_dataset = base_dataset.Dataset(
            data_dir=old_data_dir, genotype_matrix_name=old_genotype_matrix, phenotype_matrix_name=old_phenotype_matrix,
            phenotype=phenotype, datasplit=datasplit, n_outerfolds=n_outerfolds, n_innerfolds=n_innerfolds,
            test_set_size_percentage=test_set_size_percentage, val_set_size_percentage=val_set_size_percentage,
            encoding=encoding, maf_percentage=maf_perc
        )
        if not results_file_path.is_file():
            raise Exception("Results Overview file not existing. Please check: " + str(results_file_path))
        model = _model_functions.retrain_model_with_results_file(
            results_file_path=results_file_path, model_name=model_name, datasplit=datasplit,
            outerfold_number=outerfold_number, dataset=old_dataset
        )

    # Do inference and save results
    print('-----------------------------------------------')
    print("Inference on new data for " + model_name)
    y_pred_new_dataset = model.predict(X_in=new_dataset.X_full)
    final_results = pd.DataFrame(index=range(0, new_dataset.X_full.shape[0]))
    final_results.at[0:len(new_dataset.sample_ids_full) - 1, 'sample_ids'] = new_dataset.sample_ids_full.flatten()
    final_results.at[0:len(y_pred_new_dataset) - 1, 'y_pred_test'] = y_pred_new_dataset.flatten()
    print("Predictions")
    print(final_results)
    if new_phenotype_matrix is not None:
        eval_scores = \
            eval_metrics.get_evaluation_report(y_pred=y_pred_new_dataset, y_true=new_dataset.y_full, task=model.task,
                                               prefix='test_')
        print('New dataset: ')
        print(eval_scores)
        if results_file_path.is_file():
            print('Old dataset: ')
            results = pd.read_csv(results_file_path)
            results = results[results[results.columns[0]] == 'outerfold_' + str(outerfold_number)] \
                if datasplit == 'nested-cv' else results
            results = results.loc[:, [model_name in col for col in results.columns]]
            eval_dict_saved = results_analysis.result_string_to_dictionary(
                result_string=results[model_name + '___eval_metrics'][outerfold_number]
            )
            print(eval_dict_saved)
        final_results.at[0:len(new_dataset.y_full) - 1, 'y_true_test'] = new_dataset.y_full.flatten()
        for metric, value in eval_scores.items():
            final_results.at[0, metric] = value
    final_results.at[0, 'base_model_path'] = results_directory_model
    models_start_time = model_name + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    final_results.to_csv(save_dir.joinpath(
        'predict_results_on-' + new_dataset.index_file_name.split('.')[0] + '-' + models_start_time + '.csv'),
        sep=',', decimal='.', float_format='%.10f', index=False
    )
    print("Results saved")
    print("Directory: ")
    print(save_dir.joinpath('predict_results_on-' + new_dataset.index_file_name.split('.')[0] + '-' + models_start_time + '.csv'))


def retrain_on_new_data(results_directory_model: str, data_dir: str, genotype_matrix: str, phenotype_matrix: str,
                        phenotype: str, encoding: str = None, maf_percentage: int = 0, save_dir: str = None,
                        datasplit: str = 'nested-cv', n_outerfolds: int = 5, n_innerfolds: int = 5,
                        test_set_size_percentage: int = 20, val_set_size_percentage: int = 20,
                        save_final_model: bool = True):
    """
    Train a model on a new dataset using the hyperparameters that worked best for the specified model results.

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
    :param save_final_model: specify if the final model should be saved
    """

    # create Path
    results_directory_model = pathlib.Path(results_directory_model)
    if not check_functions.check_exist_directories(list_of_dirs=[results_directory_model]):
        raise Exception("See output above. Problems with specified directories")
    model_name = results_directory_model.parts[-1]
    if 'fromR' in model_name:
        import easypheno.model
        easypheno.model.__all__.extend(['_bayesfromR', 'bayesAfromR', 'bayesBfromR', 'bayesCfromR'])
    models = [model_name]
    data_dir = pathlib.Path(data_dir)
    # set save directory
    save_dir = data_dir if save_dir is None else pathlib.Path(save_dir)
    save_dir = save_dir if save_dir.is_absolute() else save_dir.resolve()
    # Check all arguments
    check_functions.check_all_specified_arguments(arguments=locals())
    # prepare all data files
    raw_data_functions.prepare_data_files(
        data_dir=data_dir, genotype_matrix_name=genotype_matrix, phenotype_matrix_name=phenotype_matrix,
        phenotype=phenotype, datasplit=datasplit, n_outerfolds=n_outerfolds, n_innerfolds=n_innerfolds,
        test_set_size_percentage=test_set_size_percentage, val_set_size_percentage=val_set_size_percentage,
        models=models, user_encoding=encoding, maf_percentage=maf_percentage
    )
    encoding = helper_functions.get_mapping_name_to_class()[model_name].standard_encoding \
        if encoding is None else encoding
    print("Load new dataset")
    new_dataset = base_dataset.Dataset(
        data_dir=data_dir, genotype_matrix_name=genotype_matrix, phenotype_matrix_name=phenotype_matrix,
        phenotype=phenotype, datasplit=datasplit, n_outerfolds=n_outerfolds, n_innerfolds=n_innerfolds,
        test_set_size_percentage=test_set_size_percentage, val_set_size_percentage=val_set_size_percentage,
        encoding=encoding, maf_percentage=maf_percentage
    )
    # train on new data
    if datasplit == 'train-val-test':
        datasplit_params = [val_set_size_percentage, test_set_size_percentage]
    elif datasplit == 'cv-test':
        datasplit_params = [n_innerfolds, test_set_size_percentage]
    elif datasplit == 'nested-cv':
        datasplit_params = [n_outerfolds, n_innerfolds]
    datasplit_subpath = helper_functions.get_subpath_for_datasplit(
        datasplit=datasplit, datasplit_params=datasplit_params
    )
    models_start_time = '+'.join(models) + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_path = save_dir.joinpath('results', genotype_matrix.split('.')[0], phenotype_matrix.split('.')[0],
                                  phenotype, datasplit + '_' + datasplit_subpath + '_MAF' + str(maf_percentage) +
                                  '_' + models_start_time)
    nested_offset = -1 if 'nested' in str(results_directory_model) else 0
    models_old = results_directory_model.parts[-2 + nested_offset].split('_')[3].split('+')
    results_file_path = results_directory_model.parents[0 - nested_offset].joinpath(
        'Results_overview_' + '_'.join(models_old) + '.csv')
    saved_outerfold_number = int(results_directory_model.parts[-2].split('_')[1]) if nested_offset != 0 else 0
    helper_functions.set_all_seeds()
    model_overview = {}
    overall_results = {}
    n_outerfolds = 1 if 'nested' not in datasplit else n_outerfolds
    for outerfold_number in range(n_outerfolds):
        print("train on new dataset")
        if 'nested' in datasplit:
            print("outerfold " + str(outerfold_number))
        model = _model_functions.retrain_model_with_results_file(
            results_file_path=results_file_path, model_name=model_name, datasplit=datasplit,
            outerfold_number=outerfold_number, dataset=new_dataset, saved_outerfold_number=saved_outerfold_number,
            saved_datasplit=results_directory_model.parts[-2 + nested_offset].split('_')[0]
        )
        if save_final_model:
            save_path = base_path.joinpath('outerfold_' + str(outerfold_number), model_name) \
                if 'nested' in datasplit else base_path.joinpath(model_name)
            if not save_path.exists():
                save_path.mkdir(parents=True)
            model.save_model(path=save_path, filename='final_retrained_model')
        outerfold_info = new_dataset.datasplit_indices['outerfold_' + str(outerfold_number)]
        X_test, y_test, sample_ids_test = \
            new_dataset.X_full[outerfold_info['test']], new_dataset.y_full[outerfold_info['test']], \
            new_dataset.sample_ids_full[outerfold_info['test']]
        X_retrain, y_retrain, sample_ids_retrain = \
            new_dataset.X_full[~np.isin(np.arange(len(new_dataset.X_full)), outerfold_info['test'])], \
            new_dataset.y_full[~np.isin(np.arange(len(new_dataset.y_full)), outerfold_info['test'])], \
            new_dataset.sample_ids_full[~np.isin(np.arange(len(new_dataset.sample_ids_full)), outerfold_info['test'])]
        y_pred_retrain = model.predict(X_in=X_retrain)
        y_pred_test = model.predict(X_in=X_test)

        # Evaluate and save results
        eval_scores = \
            eval_metrics.get_evaluation_report(y_pred=y_pred_test, y_true=y_test, task=model.task, prefix='test_')
        final_results = pd.DataFrame(index=range(0, new_dataset.y_full.shape[0]))
        final_results.at[0:len(sample_ids_retrain) - 1, 'sample_ids_retrain'] = sample_ids_retrain.flatten()
        final_results.at[0:len(y_pred_retrain) - 1, 'y_pred_retrain'] = y_pred_retrain.flatten()
        final_results.at[0:len(y_retrain) - 1, 'y_true_retrain'] = y_retrain.flatten()
        final_results.at[0:len(sample_ids_test) - 1, 'sample_ids_test'] = sample_ids_test.flatten()
        final_results.at[0:len(y_pred_test) - 1, 'y_pred_test'] = y_pred_test.flatten()
        final_results.at[0:len(y_test) - 1, 'y_true_test'] = y_test.flatten()

        for metric, value in eval_scores.items():
            final_results.at[0, metric] = value
        final_results.at[0, 'base_model_path'] = results_directory_model
        save_path = base_path.joinpath('outerfold_' + str(outerfold_number), model_name) \
            if 'nested' in datasplit else base_path.joinpath(model_name)
        if not save_path.exists():
            save_path.mkdir(parents=True)
        final_results.to_csv(save_path.joinpath('final_model_test_results.csv'),
                             sep=',', decimal='.', float_format='%.10f', index=False)
        key = 'outerfold_' + str(outerfold_number) if datasplit == 'nested-cv' else 'Test'
        best_params = model.optuna_trial.params \
            if issubclass(helper_functions.get_mapping_name_to_class()[model_name], _base_model.BaseModel) else None
        if issubclass(helper_functions.get_mapping_name_to_class()[model_name],
                      _torch_model.TorchModel) or \
                issubclass(helper_functions.get_mapping_name_to_class()[model_name],
                           _tensorflow_model.TensorflowModel):
            # additional attributes for torch and tensorflow models
            if 'n_epochs' not in best_params.keys():
                best_params['n_epochs'] = model.n_epochs
            if 'batch_size' not in best_params.keys():
                best_params['batch_size'] = model.batch_size
            best_params['early_stopping_point'] = model.early_stopping_point
        overall_results[key] = {'best_params': best_params, 'eval_metrics': eval_scores,
                                'runtime_metrics':
                                    {'process_time_mean': 0, 'process_time_std': 0,
                                     'process_time_max': 0, 'process_time_min': 0,
                                     'real_time_mean': 0, 'real_time_std': 0,
                                     'real_time_max': 0, 'real_time_min': 0}}
    model_overview[model_name] = overall_results
    nested_offset_new = -1 if 'nested' in datasplit else 0
    path_overview_file = \
        save_path.parents[0 - nested_offset_new].joinpath('Results_overview_' + '_'.join(models) + '.csv')
    helper_functions.save_model_overview_dict(model_overview=model_overview, save_path=path_overview_file)
    # generate feature importances
    feat_importance.post_generate_feature_importances(
        data_dir=str(data_dir),
        results_directory_genotype_level=str(save_dir.joinpath('results', genotype_matrix.split('.')[0]))
    )
