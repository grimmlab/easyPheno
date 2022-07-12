import numpy as np
import os
import glob
import argparse

import optuna.trial
import pandas as pd
import pathlib

from ..preprocess import base_dataset
from ..utils import helper_functions
from ..model import _base_model, _param_free_base_model
from ..evaluation import eval_metrics
from . import results_analysis


def post_generate_feature_importances(results_directory_genotype_level: str, data_dir: str):
    """
    Post-generate the feature importances for several models for all sub-folders of the specified directory of already optimized models.
    Only needed in case you e.g. forgot to implement the saving of the feature importances.

    :param results_directory_genotype_level: Results directory at the level of the name of the genotype matrix
    :param data_dir: data directory where the phenotype and genotype matrix as well as index file are stored
    """
    results_directory_genotype_level = pathlib.Path(results_directory_genotype_level)
    data_dir = pathlib.Path(data_dir)

    genotype_name = results_directory_genotype_level.parts[-1] + '.h5'
    for study in list(filter(lambda x: x.is_dir(), results_directory_genotype_level.iterdir())):
        study_name = study + '.csv'
        for phenotype in os.listdir(results_directory_genotype_level + '/' + study):
            print('++++++++++++++ PHENOTYPE ' + phenotype + ' ++++++++++++++')
            all_results_directory = results_directory_genotype_level + '/' + study + '/' + phenotype + '/'
            subdirs = \
                [name for name in os.listdir(all_results_directory) if
                 os.path.isdir(os.path.join(all_results_directory, name))]
            datasplit_maf_patterns = \
                set(['_'.join(path.split('/')[-1].split('_')[:3]) for path in subdirs])
            for pattern in list(datasplit_maf_patterns):
                maf_perc = int(pattern.split('_')[-1][3:])
                datasplit = pattern.split('_')[0]
                n_outerfolds = 5
                n_innerfolds = 5
                test_set_size_percentage = 20
                val_set_size_percentage = 20
                if datasplit == 'nested-cv':
                    n_outerfolds = int(pattern.split('_')[1].split('-')[0])
                    n_innerfolds = int(pattern.split('_')[1].split('-')[1])
                elif datasplit == 'cv-test':
                    n_innerfolds = int(pattern.split('_')[1].split('-')[0])
                    test_set_size_percentage = int(pattern.split('_')[1].split('-')[1])
                else:
                    val_set_size_percentage = int(pattern.split('_')[1].split('-')[1][:-1])
                    test_set_size_percentage = int(pattern.split('_')[1].split('-')[-1])

                dataset = base_dataset.Dataset(
                    data_dir=data_dir, genotype_matrix_name=genotype_name, phenotype_matrix_name=study_name,
                    phenotype=phenotype,
                    datasplit=datasplit, n_outerfolds=n_outerfolds, n_innerfolds=n_innerfolds,
                    test_set_size_percentage=test_set_size_percentage, val_set_size_percentage=val_set_size_percentage,
                    encoding='012', maf_percentage=maf_perc
                )
                # CAUTION: currently the '012' encoding works for all algos with featimps, may need reimplementation
                snp_ids_df = pd.DataFrame(dataset.snp_ids)
                print('Saving SNP ids')
                print(snp_ids_df.shape)
                snp_ids_df.to_csv(
                    all_results_directory + phenotype + '_snp_ids.csv',
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
                    for path in glob.glob(all_results_directory + pattern + '*'):
                        models = path.split('/')[-1].split('_')[3].split('+')
                        print('working on ' + path)
                        for current_model in models:
                            print('Model: ' + current_model)
                            if current_model in ['randomforest', 'xgboost', 'linearregression', 'elasticnet',
                                                 'bayesB', 'blup']:
                                current_directory = path + '/' + current_model + '/' if datasplit != 'nested-cv' \
                                    else path + '/outerfold_' + str(outerfold) + '/' + current_model + '/'
                                if os.path.exists(current_directory + 'final_model_feature_importances.csv'):
                                    print('Already existing')
                                    continue
                                try:
                                    results_file = glob.glob(path + '/Results_over' + '*.csv')[0]
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
                                    current_directory + 'final_model_feature_importances.csv',
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
