import numpy as np
import os
import glob

import optuna.trial
import pandas as pd
import argparse

from easyPheno.preprocess import base_dataset
from easyPheno.utils import helper_functions
from easyPheno.model import _model_functions, _base_model, _param_free_base_model
from easyPheno.evaluation import eval_metrics, results_analysis


def post_generate_feature_importances(results_directory_genotype_level: str, data_dir: str):
    """
    Summarize the results for each phenotype and datasplit for all models and save in a file.

    :param results_directory_genotype_level: Results directory at the level of the name of the genotype matrix
    """
    genotype_name = results_directory_genotype_level.split('/')[-1] + '.h5'
    for study in os.listdir(results_directory_genotype_level):
        if not os.path.isdir(results_directory_genotype_level + '/' + study):
            continue
        study_name = study + '.csv'
        for phenotype in os.listdir(results_directory_genotype_level + '/' + study):
            print('++++++++++++++ PHENOTYPE ' + phenotype + ' ++++++++++++++')
            current_directory = results_directory_genotype_level + '/' + study + '/' + phenotype + '/'
            maf_perc = 10 if 'A_thal' in results_directory_genotype_level else 0
            n_outerfolds = 3
            dataset = base_dataset.Dataset(
                data_dir=data_dir, genotype_matrix_name=genotype_name, phenotype_matrix_name=study_name,
                phenotype=phenotype if 'corn' not in results_directory_genotype_level else 'Value',
                datasplit='nested-cv', n_outerfolds=n_outerfolds, n_innerfolds=5,
                test_set_size_percentage=20, val_set_size_percentage=20,
                encoding='012', maf_percentage=maf_perc
            )
            snp_ids_df = pd.DataFrame(dataset.snp_ids)
            print('Saving SNP ids')
            print(snp_ids_df.shape)
            snp_ids_df.to_csv(
                current_directory + phenotype + '_snp_ids.csv',
                sep=',', decimal='.', float_format='%.10f',
                index=False
            )
            for outerfold in range(n_outerfolds):
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
                for path in glob.glob(current_directory + 'nested-cv*'):
                    models = path.split('/')[-1].split('_')[3].split('+')
                    print('working on ' + path)
                    for current_model in models:
                        print('Model: ' + current_model)
                        try:
                            results_file = glob.glob(path + '/*.csv')[0]
                            results = pd.read_csv(results_file)
                            results = results[results[results.columns[0]] == 'outerfold_' + str(outerfold)]
                            results = results.loc[:, [current_model in col for col in results.columns]]
                            eval_dict_saved = results_analysis.result_string_to_dictionary(
                                result_string=results[current_model + '___eval_metrics'][0]
                            )
                        except:
                            print('No results file')
                            continue
                        if current_model in ['randomforest', 'xgboost', 'linearregression', 'elasticnet',
                                             'bayesB', 'blup']:
                            current_directory = path + '/outerfold_' + str(outerfold) + '/' + current_model + '/'
                            if os.path.exists(current_directory + 'final_model_feature_importances.csv') and \
                                    current_model != 'blup':
                                print('Already existing')
                                continue
                            try:
                                modelpath = glob.glob(current_directory + '/unfitted_model*')[0].split('/')[-1]
                            except:
                                continue
                            """
                            model = _model_functions.load_retrain_model(
                                path=current_directory, filename=modelpath, X_retrain=X_retrain, y_retrain=y_retrain
                            )
                            """
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
                                    result_string=results[current_model + '___best_params'][0]
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
                                eval_metrics.get_evaluation_report(y_pred=y_pred_test, y_true=y_test, task=model.task,
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Params #
    parser.add_argument("-spec", "--species", type=str,
                        default='arabidopsis',
                        help="arabidopsis or soy or corn or simulation")
    species = vars(parser.parse_args())['species']
    if species == 'arabidopsis':
        results_directory_genotype_level = '/bit_storage/Workspace/Maura/PhenotypePred/FrontiersPaperExperiments/' \
                                           'A_thal/ld_pruned_arabidopsis_2029_maf001'
    elif species == 'soy':
         results_directory_genotype_level = '/bit_storage/Workspace/Maura/PhenotypePred/FrontiersPaperExperiments/' \
                                            'computomics/soy/genotypes_modified_soy_clean'
    elif species == 'corn':
        results_directory_genotype_level = '/bit_storage/Workspace/Maura/PhenotypePred/FrontiersPaperExperiments/' \
                                           'computomics/corn/dataset1_genotypes_modified'
    elif species == 'simulation':
        results_directory_genotype_level = '/bit_storage/Workspace/Maura/PhenotypePred/FrontiersPaperExperiments/' \
                                           'Simulation/ld_pruned_arabidopsis_10k_maf10'
    post_generate_feature_importances(
        results_directory_genotype_level=results_directory_genotype_level,
        data_dir='/bit_storage/Workspace/Maura/PhenotypePred/data/serverxchange/'

    )
