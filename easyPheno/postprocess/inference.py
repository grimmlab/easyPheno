import numpy as np
import os
import argparse

import optuna.trial
import pandas as pd
import pathlib

from ..preprocess import base_dataset
from ..utils import helper_functions
from ..model import _base_model, _param_free_base_model
from ..evaluation import eval_metrics
from . import results_analysis


def apply_final_model(path_to_model_results_folder: str, initial_data_dir: str, new_data_dir: str,
                      genotype_matrix: str, phenotype_matrix: str, phenotype: str, save_dir: str = None):
    """
    Apply a final model on a new dataset. It will be applied to the whole dataset.
    So the main purpose of this function is, if you get new samples you want to predict on.
    If the final model was saved, this will be used for inference on the new dataset.
    Otherwise, it will be retrained on the initial dataset and then used for inference on the new dataset.

    The new dataset will be filtered for the SNP ids that the model was initially trained on.

    CAUTION: the number of SNPs of the old and the new dataset has to be the same!

    :param results_directory_genotype_level:
    :param data_dir:
    :return:
    """

    results_directory_genotype_level = pathlib.Path(results_directory_genotype_level)
    data_dir = pathlib.Path(data_dir)

    genotype_name = results_directory_genotype_level.parts[-1] + '.h5'
    for phenotype_matrix in helper_functions.get_all_subdirectories_non_recursive(results_directory_genotype_level):
        study_name = phenotype_matrix.parts[-1] + '.csv'
        results_directory_phenotype_matrix_level = results_directory_genotype_level.joinpath(phenotype_matrix)
        for phenotype_folder in \
                helper_functions.get_all_subdirectories_non_recursive(results_directory_phenotype_matrix_level):
            print('++++++++++++++ PHENOTYPE ' + phenotype_folder.parts[-1] + ' ++++++++++++++')
            subdirs = [fullpath.parts[-1]
                       for fullpath in helper_functions.get_all_subdirectories_non_recursive(phenotype_folder)]
            datasplit_maf_patterns = \
                set(['_'.join(path.split('/')[-1].split('_')[:3]) for path in subdirs])
            for pattern in list(datasplit_maf_patterns):
                datasplit, n_outerfolds, n_innerfolds, val_set_size_percentage, test_set_size_percentage, maf_perc = \
                    helper_functions.get_datasplit_config_info_for_resultfolder(resultfolder=pattern)
                data_dir = str(data_dir) # TODO: delete after reconstruct
                dataset = base_dataset.Dataset(
                    data_dir=data_dir, genotype_matrix_name=genotype_name, phenotype_matrix_name=study_name,
                    phenotype=phenotype_folder.parts[-1],
                    datasplit=datasplit, n_outerfolds=n_outerfolds, n_innerfolds=n_innerfolds,
                    test_set_size_percentage=test_set_size_percentage, val_set_size_percentage=val_set_size_percentage,
                    encoding='012', maf_percentage=maf_perc
                )
                # CAUTION: currently the '012' encoding works for all algos with featimps, may need reimplementation
                snp_ids_df = pd.DataFrame(dataset.snp_ids)
                print('Saving SNP ids')
                print(snp_ids_df.shape)
                snp_ids_df.to_csv(
                    phenotype_folder.joinpath('snp_ids.csv'),
                    sep=',', decimal='.', float_format='%.10f',
                    index=False
                )
                for outerfold in range(n_outerfolds):
                    print('Working on outerfold ' + str(outerfold))
                    # Retrain on full train + val data with best hyperparams and apply on test
                    print("## Retrain best model and test ##")
                    outerfold_info = dataset.datasplit_indices['outerfold_' + str(outerfold)]
                    X_test, y_test, sample_ids_test = \
                        dataset.X_full[outerfold_info['test']], dataset.y_full[outerfold_info['test']], \
                        dataset.sample_ids_full[outerfold_info['test']]
                    X_retrain, y_retrain, sample_ids_retrain = \
                        dataset.X_full[~np.isin(np.arange(len(dataset.X_full)), outerfold_info['test'])], \
                        dataset.y_full[~np.isin(np.arange(len(dataset.y_full)), outerfold_info['test'])], \
                        dataset.sample_ids_full[~np.isin(np.arange(len(dataset.sample_ids_full)),
                                                         outerfold_info['test'])]
                    for path in phenotype_folder.glob(pattern + '*'):
                        models = path.parts[-1].split('_')[3].split('+')
                        print('working on ' + str(path))
                        for current_model in models:
                            print('Model: ' + current_model)
                            if current_model in ['randomforest', 'xgboost', 'linearregression', 'elasticnet',
                                                 'bayesB', 'blup']:
                                current_directory = path.joinpath(current_model) if datasplit != 'nested-cv' \
                                    else path.joinpath('outerfold_' + str(outerfold), current_model)
                                if os.path.exists(current_directory.joinpath('final_model_feature_importances.csv')):
                                    print('Already existing')
                                    continue
                                try:
                                    results_file = path.glob('/Results_over' + '*.csv')[0]
                                    results = pd.read_csv(results_file)
                                    results = results[results[results.columns[0]] == 'outerfold_' + str(outerfold)] \
                                        if datasplit == 'nested-cv' else results
                                    results = results.loc[:, [current_model in col for col in results.columns]]
                                    eval_dict_saved = results_analysis.result_string_to_dictionary(
                                        result_string=results[current_model + '___eval_metrics'][outerfold]
                                    )
                                except:
                                    print('No results file')
                                    continue

                                task = 'regression' if 'test_rmse' in eval_dict_saved.keys() else 'classification'
                                helper_functions.set_all_seeds()
                                if current_model in ['bayesB', 'blup']:
                                    model: _param_free_base_model.ParamFreeBaseModel = \
                                        helper_functions.get_mapping_name_to_class()[current_model](
                                            task=task,
                                        )
                                    _ = model.fit(X=X_retrain, y=y_retrain)
                                else:
                                    best_params = results_analysis.result_string_to_dictionary(
                                        result_string=results[current_model + '___best_params'][outerfold]
                                    )
                                    trial = optuna.trial.FixedTrial(params=best_params)
                                    model: _base_model.BaseModel = helper_functions.get_mapping_name_to_class()[
                                        current_model](
                                        task=task, optuna_trial=trial,
                                        n_outputs=len(np.unique(dataset.y_full)) if task == 'classification' else 1,
                                        **{}
                                    )
                                    model.retrain(X_retrain=X_retrain, y_retrain=y_retrain)
                                y_pred_test = model.predict(X_in=X_test)
                                eval_scores = \
                                    eval_metrics.get_evaluation_report(y_pred=y_pred_test, y_true=y_test,
                                                                       task=model.task,
                                                                       prefix='test_')
                                print('Compare Results from initial testing to refitting')
                                print('New fitting: ')
                                print(eval_scores)
                                print('Old fitting: ')
                                print(eval_dict_saved)
                                top_n = min(len(dataset.snp_ids), 1000)
                                feat_import_df = pd.DataFrame()
                                if current_model in ['randomforest', 'xgboost']:
                                    feature_importances = model.model.feature_importances_
                                    sorted_idx = feature_importances.argsort()[::-1][:top_n]
                                    feat_import_df['snp_ids_standard'] = dataset.snp_ids[sorted_idx]
                                    feat_import_df['feat_importance_standard'] = feature_importances[sorted_idx]
                                elif current_model in ['linearregression', 'elasticnet']:
                                    coefs = model.model.coef_
                                    dims = coefs.shape[0] if len(coefs.shape) > 1 else 1
                                    for dim in range(dims):
                                        coef = coefs[dim] if len(coefs.shape) > 1 else coefs
                                        sorted_idx = coef.argsort()[::-1][:top_n]
                                        feat_import_df['snp_ids_' + str(dim)] = dataset.snp_ids[sorted_idx]
                                        feat_import_df['coefficients_' + str(dim)] = coef[sorted_idx]
                                else:
                                    feat_imps = model.u if current_model == 'blup' else model.beta
                                    dims = 1
                                    for dim in range(dims):
                                        coef = feat_imps.flatten()
                                        sorted_idx = coef.argsort()[::-1][:top_n]
                                        feat_import_df['snp_ids_' + str(dim)] = dataset.snp_ids[sorted_idx]
                                        feat_import_df['coefficients_' + str(dim)] = coef[sorted_idx]
                                feat_import_df.to_csv(
                                    current_directory.joinpath('final_model_feature_importances.csv'),
                                    sep=',', decimal='.', float_format='%.10f',
                                    index=False
                                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dd", "--data_dir", type=str,
                        help="Provide the full path of your data directory (that contains the geno- and phenotype "
                             "files as well as the index file).")
    parser.add_argument("-rd", "--results_dir", type=str,
                        help="Provide the full path of the directory where your results are stored and for which "
                             "you want to post-generate feature importances")
    args = vars(parser.parse_args())
    data_dir = args['data_dir']
    results_directory_genotype_level = args['results_dir']

    post_generate_feature_importances(
        results_directory_genotype_level=results_directory_genotype_level, data_dir=data_dir
    )
