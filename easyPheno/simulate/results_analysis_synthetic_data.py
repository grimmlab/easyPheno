import argparse
import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt

from ..utils import helper_functions


def gather_sim_configs(sim_config_dir: pathlib.Path, save_dir: pathlib.Path):
    """
    Collect the information on the simulation configurations for all within the specified directory

    :param sim_config_dir: directory which contains the sim config files
    :param save_dir: directory to save the collected sim config info
    """
    all_sim_configs = pd.DataFrame(columns=['sim_id', 'type', 'snp_id', 'beta'])
    config_files = sim_config_dir.expanduser().glob('simulation_config_*')
    for file in config_files:
        sim_id = file.with_suffix('').name.split('_')[-1]
        background_file = pd.read_csv(sim_config_dir.joinpath('background_' + sim_id + '.csv'))
        betas_background_file = pd.read_csv(sim_config_dir.joinpath('betas_background_' + sim_id + '.csv'))
        sim_config_file = pd.read_csv(sim_config_dir.joinpath('simulation_config_' + sim_id + '.csv'))
        df_to_append = pd.DataFrame(columns=['sim_id', 'type', 'snp_id', 'beta'])
        if '-' in sim_id:
            sim_id_parts = sim_id.split('-')
            sim_ids = np.arange(int(sim_id_parts[0]), int(sim_id_parts[-1]) + 1)
        else:
            sim_ids = [int(sim_id)]
        for number in sim_ids:
            column = 'sim' + str(number)
            df_to_append['snp_id'] = background_file[column]
            df_to_append['type'] = 'background'
            df_to_append['sim_id'] = column
            df_to_append['beta'] = betas_background_file[column]
            sim_conf = sim_config_file[sim_config_file['simulation'] == number]
            for index, causal_marker in enumerate(
                    sim_conf['causal_marker'][sim_conf.index[0]][2:-2].replace('\n', '').split('\' \'')):
                row_id = df_to_append.shape[0]
                df_to_append.at[row_id, 'sim_id'] = column
                df_to_append.at[row_id, 'snp_id'] = causal_marker
                df_to_append.at[row_id, 'type'] = 'causal'
                df_to_append.at[row_id, 'beta'] = float(sim_conf['causal_beta'][sim_conf.index[0]][1:-1].split(',')[index])
            all_sim_configs = all_sim_configs.append(df_to_append, ignore_index=True)
    all_sim_configs.to_csv(save_dir.joinpath('Sim_configs_gathered_' + sim_config_dir.parts[-2] + '.csv'),
                           index=False)


def gather_feature_importances(results_dir: pathlib.Path, save_dir: pathlib.Path, datasplit_maf_pattern: str):
    """
    Collect the information on the feature importances for all models within the specified directory and for the specified datasplit maf pattern

    :param results_dir: results directory at the level of the name of the genotype matrix
    :param save_dir: directory to save the collected info
    :param datasplit_maf_pattern: datasplit maf pattern to search on
    """

    all_feat_imps = pd.DataFrame(columns=['sim_id', 'model', 'snp_id', 'feat_imp'])
    for phenotype_matrix in helper_functions.get_all_subdirectories_non_recursive(results_dir):
        results_directory_phenotype_matrix_level = results_dir.joinpath(phenotype_matrix)
        for phenotype_folder in \
                helper_functions.get_all_subdirectories_non_recursive(results_directory_phenotype_matrix_level):
            phenotype = phenotype_folder.parts[-1]
            print('++++++++++++++ PHENOTYPE ' + phenotype + ' ++++++++++++++')
            for path in phenotype_folder.glob(datasplit_maf_pattern + '*'):
                if len(list(path.glob('Results_over*.csv'))) == 0:
                    continue
                models = path.parts[-1].split('_')[3].split('+')
                print('working on ' + path)
                for current_model in models:
                    if path.joinpath('outerfold_0', current_model, 'final_model_feature_importances.csv').is_file() or \
                            path.joinpath(current_model, 'final_model_feature_importances.csv').is_file():
                        feat_imps_to_append = None
                        print('Model: ' + current_model)
                        iterations = int(datasplit_maf_pattern.split('_')[1].split('-')[0]) \
                            if 'nested' in datasplit_maf_pattern else 1
                        for outerfold_nr in range(iterations):
                            current_directory = path.joinpath('outerfold_' + str(outerfold_nr), current_model) \
                                if 'nested' in datasplit_maf_pattern else path.joinpath(current_model)
                            try:
                                feat_imp = pd.read_csv(current_directory + 'final_model_feature_importances.csv')
                                feat_imp.rename(columns={'snp_ids_0': 'snp_id'}, inplace=True)
                                feat_imp.rename(columns={'snp_ids_standard': 'snp_id'}, inplace=True)
                                if feat_imps_to_append is None:
                                    feat_imps_to_append = feat_imp.copy()
                                else:
                                    feat_imps_to_append = feat_imps_to_append.join(feat_imp.set_index('snp_id'),
                                                                                   on=['snp_id'],
                                                                                   rsuffix=current_directory.parts[-2],
                                                                                   how='outer')
                            except Exception as exc:
                                print(exc)
                        if feat_imps_to_append is not None:
                            cols_to_drop = feat_imps_to_append.columns[1:]
                            feat_imps_to_append['feat_imp'] = feat_imps_to_append.iloc[:, 1:].fillna(0).mean(axis=1)
                            feat_imps_to_append['sim_id'] = phenotype
                            feat_imps_to_append['model'] = current_model
                            feat_imps_to_append.drop(cols_to_drop, axis=1, inplace=True)
                            all_feat_imps = all_feat_imps.append(feat_imps_to_append, ignore_index=True)
    all_feat_imps.to_csv(
        save_dir.joinpath('Feature_importances_gathered_' + results_dir.parts[-1] + datasplit_maf_pattern + '.csv'),
        index=False
    )


def get_statistics_featimps_vs_simulation(all_sim_configs: pd.DataFrame, all_feat_imps: pd. Dataframe,
                                          min_perc_threshold: float = 0.01) -> pd.DataFrame:
    """
    Get statistics on feature importances compared to effect sizes on synthetic data, e.g. on how many background SNPs were detected

    :param all_sim_configs: simulation configs to consider
    :param all_feat_imps: feature importances to consider
    :param min_perc_threshold: threshold for minimum feature importance in relation to maximum feature importance for a specific model

    :return: statistics for a comparison between feature importances and effect sizes in a DataFrame
    """
    stats = pd.DataFrame(
        columns=['sim_id', 'model', '#features_with_imp', '#features_with_imp_minpercmax_' + str(min_perc_threshold),
                 '#detected_background_snps', '#detected_background_snps_minpercmax_' + str(min_perc_threshold),
                 '#background_snps',
                 'ratio_background_detected', 'ratio_background_detected_minpercmax_' + str(min_perc_threshold),
                 'ratio_background_detected_by_feat_imp',
                 'ratio_background_detected_by_feat_imp_minpercmax_' + str(min_perc_threshold),
                 '#detected_causal_snps', '#causal_snps', 'ratio_causal_detected', 'causal_snps_detected',
                 'featimp_rank_causal_snps'])
    for sim_id in set(all_sim_configs.sim_id):
        config_simid = all_sim_configs[all_sim_configs.sim_id == sim_id]
        feat_imp_simid = all_feat_imps[all_feat_imps.sim_id == sim_id]
        for model in set(feat_imp_simid.model):
            new_row = {}
            new_row['sim_id'] = sim_id
            new_row['model'] = model
            feat_imp_model = feat_imp_simid[feat_imp_simid.model == model]
            feat_imp_model_perc_threshold = feat_imp_model[
                feat_imp_model['feat_imp'].abs() > min_perc_threshold * feat_imp_model['feat_imp'].abs().max()]
            non_zero_featimps = feat_imp_model[feat_imp_model.feat_imp != 0.0]
            non_zero_featimps_perc_threshold = feat_imp_model_perc_threshold[
                feat_imp_model_perc_threshold.feat_imp != 0.0]
            new_row['#features_with_imp'] = non_zero_featimps.shape[0]
            new_row['#features_with_imp_minpercmax_' + str(min_perc_threshold)] = \
                non_zero_featimps_perc_threshold.shape[0]
            background = config_simid[config_simid.type == 'background']
            causal = config_simid[config_simid.type == 'causal']
            new_row['#detected_background_snps'] = len(
                set(non_zero_featimps.snp_id).intersection(set(background.snp_id)))
            new_row['#detected_background_snps_minpercmax_' + str(min_perc_threshold)] = len(
                set(non_zero_featimps_perc_threshold.snp_id).intersection(set(background.snp_id)))
            new_row['#background_snps'] = len(set(background.snp_id))
            new_row['ratio_background_detected_by_feat_imp'] = \
                round(new_row['#detected_background_snps'] / new_row['#features_with_imp'], 2) \
                    if new_row[ '#features_with_imp'] != 0 else np.nan
            new_row['ratio_background_detected'] = round(
                new_row['#detected_background_snps'] / new_row['#background_snps'], 2)
            new_row['ratio_background_detected_by_feat_imp_minpercmax_' + str(min_perc_threshold)] = \
                round(new_row['#detected_background_snps_minpercmax_' + str(min_perc_threshold)] /
                      new_row['#features_with_imp_minpercmax_' + str(min_perc_threshold)], 2) \
                    if new_row['#features_with_imp_minpercmax_' + str(min_perc_threshold)] != 0 else np.nan
            new_row['ratio_background_detected_minpercmax_' + str(min_perc_threshold)] = round(
                new_row['#detected_background_snps_minpercmax_' + str(min_perc_threshold)] / new_row['#background_snps'], 2)
            new_row['#detected_causal_snps'] = len(set(non_zero_featimps.snp_id).intersection(set(causal.snp_id)))
            new_row['#causal_snps'] = len(set(causal.snp_id))
            new_row['ratio_causal_detected'] = round(new_row['#detected_causal_snps'] / new_row['#causal_snps'], 2)
            new_row['causal_snps_detected'] = 'Yes' if new_row['ratio_causal_detected'] == 1 else 'No'
            featimp_sorted = non_zero_featimps.sort_values(by='feat_imp', ascending=False).reset_index(drop=True)
            new_row['featimp_rank_causal_snps'] = list(
                featimp_sorted[featimp_sorted.snp_id.isin(set(causal.snp_id))].index + 1)
            stats = stats.append(new_row, ignore_index=True)
    stats = stats.sort_values(by=['sim_id', 'model'])
    return stats


def generate_scatterplots_featimps_vs_simulation(all_feat_imps: pd.DataFrame, all_sim_configs: pd.DataFrame,
                                                 save_dir: pathlib.Path, datasplit_maf_pattern: str):
    """
    Generate scatterplots based on feature importances and effect sizes on synthetic data. One plot containing all models for the specified datasplit maf pattern as well as single plots for each model will be generated and saved.

    :param all_sim_configs: simulation configs to consider
    :param all_feat_imps: feature importances to consider
    :param save_dir: directory to save the plots
    :param datasplit_maf_pattern: datasplit maf pattern to search on
    """
    sim_ids = list(all_feat_imps['sim_id'])
    sim_infos = all_sim_configs[all_sim_configs['sim_id'].isin(sim_ids)]
    beta_max = sim_infos[sim_infos['beta'] != 0]['beta'].astype(float).abs().max()
    models_total = list(all_sim_configs['model'])
    for models in [models_total, [single_model for single_model in models_total]]:
        rows = np.ceil(sim_ids / 3)
        fig = plt.figure(figsize=(rows * 3, 16))
        for plot_nr, sim_id in enumerate(sim_ids):
            ax = fig.add_subplot(rows, 3, plot_nr + 1)
            sim_info_for_id = sim_infos[sim_infos['sim_id'] == sim_id]
            feat_info_for_id = all_feat_imps[all_feat_imps['sim_id'] == sim_id]
            for model in models:
                feat_info_for_model = feat_info_for_id[feat_info_for_id['model'] == model]
                plot_info = sim_info_for_id.iloc[:, 1:].join(feat_info_for_model.iloc[:, 2:].set_index('snp_id'),
                                                             on=['snp_id']).fillna(0)
                plot_info_non_zero = plot_info[plot_info['feat_imp'] != 0]
                corr = plot_info_non_zero['beta'].astype(float).corr(plot_info_non_zero['feat_imp'].astype(float),
                                                                     method="pearson")
                plot_info_non_zero['beta'] = plot_info_non_zero['beta'].astype(float).abs()
                plot_info_non_zero['feat_imp'] = plot_info_non_zero['feat_imp'].astype(float).abs()
                plot_info_non_zero['feat_imp'] = \
                    (plot_info_non_zero['feat_imp'] - plot_info_non_zero['feat_imp'].min()) / \
                    (plot_info_non_zero['feat_imp'].max() - plot_info_non_zero['feat_imp'].min())
                plot_info_background = plot_info_non_zero[plot_info_non_zero['type'] == 'background']
                plot_info_causal = plot_info_non_zero[plot_info_non_zero['type'] == 'causal']
                alpha_offset = 0.2 if len(models) == 1 else 0
                ax.scatter(x=plot_info_background['feat_imp'], y=plot_info_background['beta'],
                           alpha=0.3 + alpha_offset, label=model + '(' + str(round(corr, 2)) + ')')
                ax.scatter(x=plot_info_causal['feat_imp'], y=plot_info_causal['beta'],
                           alpha=0.7, label='_nolegend_', edgecolors='black')
            ax.set(xscale="log", yscale="log")
            ax.set_xlim(0.00001, 1 + np.exp(0.2))
            ax.set_ylim(0.00001, beta_max + 2)
            ax.set_xlabel('Feature importance', fontsize=10)
            ax.set_ylabel('Effect size', fontsize=10)
            ax.set_title(str(sim_id), fontsize=12)
            ax.legend(loc='lower right', fontsize=10, labelspacing=0.2, handletextpad=0.5, borderaxespad=0.2)
            ax.tick_params(labelsize=10)
        plt.subplots_adjust(hspace=0.1)
        suptitle = 'Feature importances of all models versus effect sizes' if len(models) > 1 \
            else 'Feature importances of ' + model + ' versus effect sizes'
        fig.suptitle(suptitle, fontsize=13)
        fig.tight_layout()
        model_string = models[0] if len(models) == 1 else 'all'
        plt.savefig(
            save_dir.joinpath('Scatterplot_simulated_effect_sizes_vs_featimps_statistics_' + model_string + '_' +
                              datasplit_maf_pattern + '.pdf'), bbox_inches='tight', dpi=300)


def featimps_vs_simulation(results_directory_genotype_level: str, sim_config_dir: str, save_dir: str):
    """
    Analyze feature importances versus effect sizes on synthetic data, both by retrieving stastistics and generating plots

    :param results_directory_genotype_level: results directory at the level of the name of the genotype matrix
    :param sim_config_dir: directory which contains the sim config files
    :param save_dir: directory to save the results
    """

    results_directory_genotype_level = pathlib.Path(results_directory_genotype_level)
    sim_config_dir = pathlib.Path(sim_config_dir)
    save_dir = pathlib.Path(save_dir) if save_dir is not None else results_directory_genotype_level
    sim_configs_gathered_path = save_dir.joinpath('Sim_configs_gathered_' + sim_config_dir.parts[-2] + '.csv')
    if not sim_configs_gathered_path.is_file():
        print('Prepare sim config info')
        gather_sim_configs(sim_config_dir=sim_config_dir, save_dir=save_dir)

    for phenotype_matrix in helper_functions.get_all_subdirectories_non_recursive(results_directory_genotype_level):
        results_directory_phenotype_matrix_level = results_directory_genotype_level.joinpath(phenotype_matrix)
        for phenotype_folder in \
                helper_functions.get_all_subdirectories_non_recursive(results_directory_phenotype_matrix_level):
            subdirs = [fullpath.parts[-1]
                       for fullpath in helper_functions.get_all_subdirectories_non_recursive(phenotype_folder)]
            datasplit_maf_patterns = set(['_'.join(path.split('_')[:3]) for path in subdirs])

    for datasplit_maf_pattern in datasplit_maf_patterns:
        print('Simulation vs Feature Importances for ' + datasplit_maf_pattern)
        feat_imps_gathered_path = save_dir.joinpath('Feature_importances_gathered_' +
                                                    results_directory_genotype_level.parts[-1] +
                                                    datasplit_maf_pattern + '.csv')
        if not feat_imps_gathered_path.is_file():
            print('Prepare feature importance info')
            gather_feature_importances(results_dir=results_directory_genotype_level, save_dir=save_dir,
                                       datasplit_maf_pattern=datasplit_maf_pattern)
        all_sim_configs = pd.read_csv(sim_configs_gathered_path)
        all_feat_imps = pd.read_csv(feat_imps_gathered_path)
        print('Generate and save statistics')
        stats = get_statistics_featimps_vs_simulation(all_sim_configs=all_sim_configs, all_feat_imps=all_feat_imps)
        stats.to_csv(save_dir.joinpath('Simulated_effect_sizes_vs_featimps_statistics_' +
                                       datasplit_maf_pattern + '.csv'))
        print('Generate and save plots')
        generate_scatterplots_featimps_vs_simulation(all_sim_configs=all_sim_configs, all_feat_imps=all_feat_imps,
                                                     save_dir=save_dir, datasplit_maf_pattern=datasplit_maf_pattern)


if __name__ == "__main__":
    """
    Run file to generate statistics and plots of feature importances versus effect sizes on synthetic data
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-rd", "--results_dir", type=str,
                        help="Provide the full path of the directory where your results are stored "
                             "(name of the genotype matrix level)")
    parser.add_argument("-simd", "--sim_config_dir", type=str,
                        help="Provide the full path of the directory where the simulation configs are stored")
    parser.add_argument("-sd", "--save_dir", type=str, default=None,
                        help="Define save directory for the plots. Default is the same as results directory.")

    args = vars(parser.parse_args())
    results_directory_genotype_level = args['results_dir']
    sim_config_dir = args['sim_config_dir']
    save_dir = args['save_dir']

    featimps_vs_simulation(
        results_directory_genotype_level=results_directory_genotype_level,
        sim_config_dir=sim_config_dir, save_dir=save_dir
    )
