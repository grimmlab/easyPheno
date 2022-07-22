import numpy as np
import h5py
import pandas as pd
import pathlib
import datetime

from ..preprocess import base_dataset, raw_data_functions
from ..utils import helper_functions, check_functions
from ..model import _model_functions
from ..evaluation import eval_metrics
from . import feat_importance


def apply_final_model(results_directory_model: str, old_data_dir: str, new_data_dir: str,
                      new_genotype_matrix: str, new_phenotype_matrix: str, save_dir: str = None):
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
    :param new_phenotype_matrix: new phenotype matrix (incl. file suffix)
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
    datasplit, n_outerfolds, n_innerfolds, val_set_size_percentage, test_set_size_percentage, maf_perc = \
        helper_functions.get_datasplit_config_info_for_resultfolder(resultfolder=result_folder_name)
    nested_offset = -1 if 'nested' in datasplit else 0
    phenotype = results_directory_model.parts[-3 + nested_offset]
    old_genotype_matrix = results_directory_model.parts[-5 + nested_offset]
    old_phenotype_matrix = results_directory_model.parts[-4 + nested_offset]
    if not check_functions.check_exist_files(
            list_of_files=[
                new_data_dir.joinpath(new_phenotype_matrix), new_data_dir.joinpath(new_genotype_matrix),
                old_data_dir.joinpath(base_dataset.Dataset.get_index_file_name(
                    genotype_matrix_name=old_genotype_matrix, phenotype_matrix_name=old_phenotype_matrix,
                    phenotype=phenotype))]):
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
        old_dataset_snp_ids = f[f'matched_data/final_snp_ids/{encoding}/maf_{maf_perc}_snp_ids'][:]
    if not check_functions.compare_snp_id_vectors(snp_id_vector_big_equal=new_dataset.snp_ids,
                                                  snp_id_vector_small_equal=old_dataset_snp_ids):
        raise Exception('SNPids of initial dataset and new dataset do not match.')
    old_dataset_snp_ids = np.asarray(old_dataset_snp_ids.index, dtype=new_dataset.snp_ids.dtype).flatten()
    _, ids_to_keep = \
        (np.reshape(old_dataset_snp_ids, (old_dataset_snp_ids.shape[0], 1)) == new_dataset.snp_ids).nonzero()
    new_dataset.X_full = new_dataset.X_full[:, ids_to_keep]

    # Prepare the model
    full_model_path = results_directory_model.joinpath("final_retrained_model")
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
        outerfold_number = int(results_directory_model.parent.split('_')[1]) if 'nested' in datasplit else 0
        models = results_directory_model.parts[-2+nested_offset].split('_')[3].split('+')
        results_file_path = \
            results_directory_model.parents[0-nested_offset].joinpath('Results_overview_' + '_'.join(models) + '.csv')
        if not results_file_path.is_file():
            raise Exception("Results Overview file not existing. Please check: " + str(results_file_path))
        model = _model_functions.retrain_model_with_results_file(
            results_file_path=results_file_path, model_name=model_name, datasplit=datasplit,
            outerfold_number=outerfold_number, dataset=old_dataset
        )

    # Do inference and save results
    print("Inference on new data")
    y_pred_new_dataset = model.predict(X_in=new_dataset.X_full)
    eval_scores = \
        eval_metrics.get_evaluation_report(y_pred=y_pred_new_dataset, y_true=new_dataset.y_full, task=model.task,
                                           prefix='test_')
    print(eval_scores)
    final_results = pd.DataFrame(index=range(0, new_dataset.y_full.shape[0]))
    final_results.at[0:len(new_dataset.sample_ids_full) - 1, 'sample_ids'] = new_dataset.sample_ids_full.flatten()
    final_results.at[0:len(y_pred_new_dataset) - 1, 'y_pred_test'] = y_pred_new_dataset.flatten()
    final_results.at[0:len(new_dataset.y_full) - 1, 'y_true_test'] = new_dataset.y_full.flatten()
    for metric, value in eval_scores.items():
        final_results.at[0, metric] = value
    final_results.at[0, 'base_model_path'] = results_directory_model
    final_results.to_csv(
        save_dir.joinpath('predict_results_on_' + new_dataset.index_file_name.split('.')[0] + '.csv'),
        sep=',', decimal='.', float_format='%.10f', index=False
    )


def retrain_on_new_data(path_to_model_results_folder: str,
                        data_dir: str, genotype_matrix: str, phenotype_matrix: str, phenotype: str,
                        encoding: str = None, maf_percentage: int = 0, save_dir: str = None,
                        datasplit: str = 'nested-cv', n_outerfolds: int = 5, n_innerfolds: int = 5,
                        test_set_size_percentage: int = 20, val_set_size_percentage: int = 20,
                        save_final_model: bool = False,
                        batch_size: int = 32, n_epochs: int = 100000):

    # create Path
    path_to_model_results_folder = pathlib.Path(path_to_model_results_folder)
    if not check_functions.check_exist_directories(list_of_dirs=[path_to_model_results_folder]):
        raise Exception("See output above. Problems with specified directories")
    model_name = path_to_model_results_folder.parts[-1]
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
    helper_functions.set_all_seeds()
    for outerfold_number in range(n_outerfolds):
        print("train on new dataset")
        if 'nested' in datasplit:
            print("outerfold " + str(outerfold_number))
        model = _model_functions.retrain_model_with_results_file(
            results_file_path=path_to_model_results_folder, model_name=model_name, datasplit=datasplit,
            outerfold_number=outerfold_number, dataset=new_dataset
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
        final_results.at[0, 'base_model_path'] = path_to_model_results_folder
        save_path = base_path.joinpath('outerfold_' + str(outerfold_number), model_name) \
            if 'nested' in datasplit else base_path.joinpath(model_name)
        if not save_path.exists():
            save_path.mkdir(parents=True)
        final_results.to_csv(save_path.joinpath('final_model_test_results.csv'),
                             sep=',', decimal='.', float_format='%.10f', index=False)
    # generate feature importances
    feat_importance.post_generate_feature_importances(
        data_dir=str(data_dir),
        results_directory_genotype_level=str(save_dir.joinpath('results', genotype_matrix.split('.')[0]))
    )
