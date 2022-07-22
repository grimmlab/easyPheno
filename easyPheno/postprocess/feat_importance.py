import os

import pandas as pd
import pathlib

from ..preprocess import base_dataset
from ..utils import helper_functions, check_functions
from ..model import _model_functions


def post_generate_feature_importances(results_directory_genotype_level: str, data_dir: str):
    """
    Post-generate the feature importances for several models for all sub-folders of the specified directory of already optimized models.
    Only needed in case you e.g. forgot to implement the saving of the feature importances.

    :param results_directory_genotype_level: results directory at the level of the name of the genotype matrix
    :param data_dir: data directory where the phenotype and genotype matrix as well as index file are stored
    """
    results_directory_genotype_level = pathlib.Path(results_directory_genotype_level)
    data_dir = pathlib.Path(data_dir)

    # Check user inputs
    print("Checking user inputs")
    if not check_functions.check_exist_directories(list_of_dirs=[results_directory_genotype_level, data_dir]):
        raise Exception("See output above. Problems with specified directories")
    if not data_dir.joinpath(results_directory_genotype_level.parts[-1] + '.h5').is_file():
        raise Exception("Genotype matrix specified does not exist: " + str(results_directory_genotype_level) +
                        "\n Make sure the results directory is at the level fo the genotype matrix name.")

    genotype_name = results_directory_genotype_level.parts[-1] + '.h5'
    for phenotype_matrix in helper_functions.get_all_subdirectories_non_recursive(results_directory_genotype_level):
        study_name = phenotype_matrix.parts[-1] + '.csv'
        for phenotype_folder in \
                helper_functions.get_all_subdirectories_non_recursive(phenotype_matrix):
            print('++++++++++++++ PHENOTYPE ' + phenotype_folder.parts[-1] + ' ++++++++++++++')
            subdirs = [fullpath.parts[-1]
                       for fullpath in helper_functions.get_all_subdirectories_non_recursive(phenotype_folder)]
            datasplit_maf_patterns = set(['_'.join(path.split('_')[:3]) for path in subdirs])
            for pattern in list(datasplit_maf_patterns):
                datasplit, n_outerfolds, n_innerfolds, val_set_size_percentage, test_set_size_percentage, maf_perc = \
                    helper_functions.get_datasplit_config_info_for_resultfolder(resultfolder=pattern)
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
                snp_ids_df.to_csv(
                    phenotype_folder.joinpath('snp_ids.csv'),
                    sep=',', decimal='.', float_format='%.10f',
                    index=False
                )

                for outerfold in range(n_outerfolds):
                    print('Working on outerfold ' + str(outerfold))
                    # Retrain on full train + val data with best hyperparams and apply on test
                    print("## Retrain best model and test ##")
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
                                results_file_path = path.joinpath('Results_overview_' + '_'.join(models) + '.csv')
                                if not results_file_path.is_file():
                                    print('No results file')
                                    continue
                                model = _model_functions.retrain_model_with_results_file(
                                    results_file_path=results_file_path, model_name=current_model, datasplit=datasplit,
                                    outerfold_number=outerfold, dataset=dataset
                                )
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
