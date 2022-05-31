import numpy as np
import os
import glob

import optuna.trial
import pandas as pd

from easyPheno.preprocess import base_dataset
from easyPheno.utils import helper_functions
from easyPheno.model import _model_functions, _base_model
from easyPheno.evaluation import eval_metrics, results_analysis


def post_generate_feature_importances(results_directory_genotype_level: str, data_dir: str):
    """
    Summarize the results for each phenotype and datasplit for all models and save in a file.

    :param results_directory_genotype_level: Results directory at the level of the name of the genotype matrix
    """
    genotype_name = results_directory_genotype_level.split('/')[-1] + '.h5'
    for study in os.listdir(results_directory_genotype_level):
        study_name = study + '.csv'
        for phenotype in os.listdir(results_directory_genotype_level + '/' + study):
            print('++++++++++++++ PHENOTYPE ' + phenotype + ' ++++++++++++++')
            current_directory = results_directory_genotype_level + '/' + study + '/' + phenotype + '/'
            dataset = base_dataset.Dataset(
                data_dir=data_dir, genotype_matrix_name=genotype_name, phenotype_matrix_name=study_name,
                phenotype=phenotype, datasplit='cv-test', n_outerfolds=5, n_innerfolds=5,
                test_set_size_percentage=20, val_set_size_percentage=20,
                encoding='012', maf_percentage=10
            )
            # Retrain on full train + val data with best hyperparams and apply on test
            print("## Retrain best model and test ##")
            outerfold_info = dataset.datasplit_indices['outerfold_0']
            X_test, y_test, sample_ids_test = \
                dataset.X_full[outerfold_info['test']], dataset.y_full[outerfold_info['test']], \
                dataset.sample_ids_full[outerfold_info['test']]
            X_retrain, y_retrain, sample_ids_retrain = \
                dataset.X_full[~np.isin(np.arange(len(dataset.X_full)), outerfold_info['test'])], \
                dataset.y_full[~np.isin(np.arange(len(dataset.y_full)), outerfold_info['test'])], \
                dataset.sample_ids_full[~np.isin(np.arange(len(dataset.sample_ids_full)),
                                                      outerfold_info['test'])]
            snp_ids_df = pd.DataFrame(dataset.snp_ids)
            print('Saving SNP ids')
            print(snp_ids_df.shape)
            snp_ids_df.to_csv(
                current_directory + phenotype + '_snp_ids.csv',
                sep=',', decimal='.', float_format='%.10f',
                index=False
            )
            for path in glob.glob(current_directory + '*'):
                models = path.split('/')[-1].split('_')[3:-2]
                print('working on ' + path)
                for current_model in models:
                    print('Model: ' + current_model)
                    try:
                        results_file = glob.glob(path + '/*.csv')[0]
                        results = pd.read_csv(results_file)
                        results = results.loc[:, [current_model in col for col in results.columns]]
                        eval_dict_saved = results_analysis.result_string_to_dictionary(
                            result_string=results[current_model + '___eval_metrics'][0]
                        )
                    except:
                        print('No results file')
                        continue
                    if current_model in ['randomforest', 'xgboost', 'linearregression', 'elasticnet']:
                        current_directory = path + '/' + current_model + '/'
                        if os.path.exists(current_directory + 'final_model_feature_importances.csv'):
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
                        best_params = results_analysis.result_string_to_dictionary(
                            result_string=results[current_model + '___best_params'][0]
                        )
                        task = 'regression' if 'test_rmse' in eval_dict_saved.keys() else 'classification'
                        trial = optuna.trial.FixedTrial(params=best_params)
                        helper_functions.set_all_seeds()
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
                        else:
                            coefs = model.model.coef_
                            dims = coefs.shape[0] if len(coefs.shape) > 1 else 1
                            for dim in range(dims):
                                coef = coefs[dim] if len(coefs.shape) > 1 else coefs
                                sorted_idx = coef.argsort()[::-1][:top_n]
                                feat_import_df['snp_ids_' + str(dim)] = dataset.snp_ids[sorted_idx]
                                feat_import_df['coefficients_' + str(dim)] = coef[sorted_idx]
                        feat_import_df.to_csv(
                            current_directory + 'final_model_feature_importances.csv',
                            sep=',', decimal='.', float_format='%.10f',
                            index=False
                        )


post_generate_feature_importances(
    results_directory_genotype_level=
    '/bit_storage/Workspace/Maura/PhenotypePred/FrontiersPaperExperiments/A_thal/ld_pruned_arabidopsis_2029_maf001',
    data_dir='/bit_storage/Workspace/Maura/PhenotypePred/data/ArabidopsisThaliana/'
)
