import numpy as np
import os
import argparse
import h5py
import optuna.trial
import pandas as pd
import pathlib

from ..preprocess import base_dataset, raw_data_functions
from ..utils import helper_functions, check_functions
from ..model import _base_model, _param_free_base_model, _model_functions
from ..evaluation import eval_metrics
from . import results_analysis


def apply_final_model(results_directory_model: str, old_data_dir: str, new_data_dir: str,
                      new_genotype_matrix: str, new_phenotype_matrix: str, new_phenotype: str, save_dir: str = None):
    """
    Apply a final model on a new dataset. It will be applied to the whole dataset.
    So the main purpose of this function is, if you get new samples you want to predict on.
    If the final model was saved, this will be used for inference on the new dataset.
    Otherwise, it will be retrained on the initial dataset and then used for inference on the new dataset.

    The new dataset will be filtered for the SNP ids that the model was initially trained on.

    CAUTION: the SNPs of the old and the new dataset have to be the same!

    :param results_directory_model: directory that contains the model results that you want to use
    :param old_data_dir: directory that contains the data that the model was trained on if it was not saved
    :param new_data_dir: directory that contains the new genotype and phenotype matrix
    :param new_genotype_matrix: new genotype matrix (incl. file suffix)
    :param new_phenotype_matrix: new phenotype matrix (incl. file suffix)
    :param new_phenotype: new phenotype to predict on
    :param save_dir: directory to store the results
    """
    results_directory_model = pathlib.Path(results_directory_model)
    old_data_dir = pathlib.Path(old_data_dir) if old_data_dir is not None else None
    new_data_dir = pathlib.Path(new_data_dir)
    save_dir = pathlib.Path(save_dir)

    # Check user inputs
    print("Checking user inputs")
    full_model_path = results_directory_model.joinpath("final_retrained_model")
    if not full_model_path.is_file() and old_data_dir is None:
        raise Exception("Final model was not saved. "
                        "Please provide directory containing the data that the model was initially trained on.")
    dirs_to_check = [results_directory_model, new_data_dir, save_dir]
    if old_data_dir is not None:
        dirs_to_check.append(old_data_dir)
    if not check_functions.check_exist_directories(list_of_dirs=dirs_to_check):
        raise Exception("See output above. Problems with specified directories")
    if not check_functions.check_exist_files(
            list_of_files=[new_data_dir.joinpath(new_phenotype_matrix), new_data_dir.joinpath(new_genotype_matrix)]):
        raise Exception("See output above. Problems with specified files.")

    # Prepare the new data
    print("Preparing the new dataset")
    datasplit_maf_pattern = results_directory_model.parts[-3] if 'nested' in str(results_directory_model) \
        else results_directory_model.parts[-2]
    model_name = results_directory_model.parts[-1]
    datasplit, n_outerfolds, n_innerfolds, val_set_size_percentage, test_set_size_percentage, maf_perc = \
        helper_functions.get_datasplit_config_info_for_resultfolder(resultfolder=datasplit_maf_pattern)
    raw_data_functions.prepare_data_files(
        data_dir=new_data_dir, genotype_matrix_name=new_genotype_matrix, phenotype_matrix_name=new_phenotype_matrix,
        phenotype=new_phenotype, datasplit=datasplit, n_outerfolds=n_outerfolds, n_innerfolds=n_innerfolds,
        test_set_size_percentage=test_set_size_percentage, val_set_size_percentage=val_set_size_percentage,
        models=[model_name], user_encoding=None, maf_percentage=maf_perc
    )
    encoding = helper_functions.get_mapping_name_to_class()[model_name].standard_encoding
    new_dataset = base_dataset.Dataset(
        data_dir=new_data_dir, genotype_matrix_name=new_genotype_matrix, phenotype_matrix_name=new_phenotype_matrix,
        phenotype=new_phenotype, datasplit=datasplit, n_outerfolds=n_outerfolds, n_innerfolds=n_innerfolds,
        test_set_size_percentage=test_set_size_percentage, val_set_size_percentage=val_set_size_percentage,
        encoding=encoding, maf_percentage=maf_perc, do_snp_filters=False
    )

    # Check and filter SNPids in comparison with old dataset
    nested_offset = -1 if 'nested' in datasplit_maf_pattern else 0
    old_genotype_matrix = results_directory_model.parts[-5 + nested_offset]
    old_phenotype_matrix = results_directory_model.parts[-4 + nested_offset]
    old_phenotype = results_directory_model.parts[-3 + nested_offset]
    with h5py.File(new_data_dir.joinpath(
            base_dataset.Dataset.get_index_file_name(genotype_matrix_name=old_genotype_matrix,
                                                     phenotype_matrix_name=old_phenotype_matrix,
                                                     phenotype=old_phenotype)), "r") as f:
        old_dataset_snp_ids = f[f'matched_data/final_snp_ids/{encoding}/maf_{maf_perc}_snp_ids'][:]
    if not check_functions.compare_snp_id_vectors(snp_id_vector_big_equal=new_dataset.snp_ids,
                                                  snp_id_vector_small_equal=old_dataset_snp_ids):
        raise Exception('SNPids of initial dataset and new dataset do not match.')
    old_dataset_snp_ids = np.asarray(old_dataset_snp_ids.index, dtype=new_dataset.snp_ids.dtype).flatten()
    _, ids_to_keep = \
        (np.reshape(old_dataset_snp_ids, (old_dataset_snp_ids.shape[0], 1)) == new_dataset.snp_ids).nonzero()
    new_dataset.X_full = new_dataset.X_full[:, ids_to_keep]

    # Prepare the model
    if full_model_path.is_file():
        print("Loading saved model")
        model = _model_functions.load_model(path=results_directory_model, filename=full_model_path.parts[-1])
    else:
        print("Retraining model")
        print("Loading old dataset")
        old_dataset = base_dataset.Dataset(
            data_dir=old_data_dir, genotype_matrix_name=old_genotype_matrix, phenotype_matrix_name=old_phenotype_matrix,
            phenotype=old_phenotype,
            datasplit=datasplit, n_outerfolds=n_outerfolds, n_innerfolds=n_innerfolds,
            test_set_size_percentage=test_set_size_percentage, val_set_size_percentage=val_set_size_percentage,
            encoding=encoding, maf_percentage=maf_perc
        )
        outerfold_number = int(results_directory_model.parent.split('_')[1]) if 'nested' in datasplit_maf_pattern else 0
        results_file_path = list(results_directory_model.parents[0 - nested_offset].glob('Results*.csv'))[0]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-odd", "--old_data_dir", type=str, default=None,
                        help="Provide the full path of the old data directory (that contains the geno- and phenotype "
                             "files as well as the index file the model was trained on). "
                             "Only needed if the final model was not saved.")
    parser.add_argument("-ndd", "--new_data_dir", type=str,
                        help="Provide the full path of the new data directory that contains the geno- and phenotype "
                             "files you want to predict on")
    parser.add_argument("-ngm", "--new_genotype_matrix", type=str,
                        help="Provide the name of the new genotype matrix you want to predict on")
    parser.add_argument("-npm", "--new_phenotype_matrix", type=str,
                        help="Provide the name of the new phenotype matrix you want to predict on")
    parser.add_argument("-np", "--new_phenotype", type=str,
                        help="Provide the name of the new phenotype you want to predict on")
    parser.add_argument("-sd", "--save_dir", type=str,
                        help="Define the save directory for the results.")
    parser.add_argument("-rd", "--results_dir_model", type=str,
                        help="Provide the full path of the directory where your results of the model "
                             "you want to use are stored")
    args = vars(parser.parse_args())
    old_data_dir = args['old_data_dir']
    new_data_dir = args['new_data_dir']
    new_genotype_matrix = args['new_genotype_matrix']
    new_phenotype_matrix = args['new_phenotype_matrix']
    new_phenotype = args["new_phenotype"]
    save_dir = args["save_dir"]
    results_directory_model = args['results_dir_model']

    apply_final_model(
        results_directory_model=results_directory_model, old_data_dir=old_data_dir, new_data_dir=new_data_dir,
        new_genotype_matrix=new_genotype_matrix, new_phenotype_matrix=new_phenotype_matrix, new_phenotype=new_phenotype,
        save_dir=save_dir
    )
