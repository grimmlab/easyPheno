import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

from ..utils import helper_functions, check_functions


def summarize_results_per_phenotype_and_datasplit(results_directory_genotype_level: str):
    """
    Summarize the results for each phenotype and datasplit for all models and save in a file.

    The following files will be created:

        - at phenotype-folder level within results directories:

            - Detailed_results_summary_*PHENOTYPE*DATASPLIT-PATTERN*.xlsx: .xlsx-file containing detailed results for each phenotype and datasplit-maf pattern (e.g. with all runtime results etc.)
            - Results_summary**PHENOTYPE*DATASPLIT-PATTERN*.csv: .csv-file containing an overview of the performance of each model for a phenotype and datasplit-maf pattern combination

        - at genotype-folder level within results directories (the one that was specified):

            - Results_summary_all_phenotypes*DATASPLIT-PATTERN*.xlsx: .xlsx-file containing an overview of the performance of each model on each phenotype used for this genotype matrix with the specified datasplit-maf pattern
            - Results_summary_all_phenotypes*DATASPLIT-PATTERN*.csv: only overview sheet of Results_summary*DATASPLIT-PATTERN*.xlsx

    :param results_directory_genotype_level: results directory at the level of the name of the genotype matrix
    """
    results_directory_genotype_level = pathlib.Path(results_directory_genotype_level)

    # Check user inputs
    print("Checking user inputs")
    if not check_functions.check_exist_directories(list_of_dirs=[results_directory_genotype_level]):
        raise Exception("See output above. Problems with specified directories")
    if not results_directory_genotype_level.parts[-2] == 'results':
        raise Exception("Problems with specified directory: " + str(results_directory_genotype_level) +
                        "\n Make sure the results directory is at the level fo the genotype matrix name.")

    for phenotype_matrix in helper_functions.get_all_subdirectories_non_recursive(results_directory_genotype_level):
        results_directory_phenotype_matrix_level = results_directory_genotype_level.joinpath(phenotype_matrix)
        for phenotype_folder in \
                helper_functions.get_all_subdirectories_non_recursive(results_directory_phenotype_matrix_level):
            phenotype = phenotype_folder.parts[-1]
            print('++++++++++++++ PHENOTYPE ' + phenotype + ' ++++++++++++++')
            subdirs = [fullpath.parts[-1]
                       for fullpath in helper_functions.get_all_subdirectories_non_recursive(phenotype_folder)]
            datasplit_maf_patterns = set(['_'.join(path.split('_')[:3]) for path in subdirs])
            for pattern in list(datasplit_maf_patterns):
                writer = pd.ExcelWriter(
                    phenotype_folder.joinpath('Detailed_results_summary_' + phenotype + '_' + pattern + '.xlsx'),
                    engine='xlsxwriter'
                )
                overview_df = None
                print('----- Datasplit pattern ' + pattern + ' -----')
                print('Got results for ' + str(len([model for path in phenotype_folder.glob(pattern + '*')
                                                    for model in path.parts[-1].split('_')[3].split('+')])) +
                      ' models.')
                for path in phenotype_folder.glob(pattern + '*'):
                    models = path.parts[-1].split('_')[3].split('+')
                    for current_model in models:
                        print('### Results for ' + current_model + ' ###')
                        try:
                            results_file = list(path.glob('Results*.csv'))[0]
                            results = pd.read_csv(results_file)
                            results = results.loc[:, [current_model in col for col in results.columns]]
                            if 'nested' in pattern:
                                eval_dict_std = result_string_to_dictionary(
                                    result_string=results[current_model + '___eval_metrics'].iloc[-1])
                                eval_dict = result_string_to_dictionary(
                                    result_string=results[current_model + '___eval_metrics'].iloc[-2]
                                )
                                runtime_dict_std = result_string_to_dictionary(
                                    result_string=results[current_model + '___runtime_metrics'].iloc[-1]
                                )
                                runtime_dict = result_string_to_dictionary(
                                    result_string=results[current_model + '___runtime_metrics'].iloc[-2]
                                )
                            else:
                                eval_dict = result_string_to_dictionary(
                                    result_string=results[current_model + '___eval_metrics'][0]
                                )
                                runtime_dict = result_string_to_dictionary(
                                    result_string=results[current_model + '___runtime_metrics'][0]
                                )
                            results.to_excel(writer, sheet_name=current_model + '_results', index=False)
                            eval_dict = {str(key) + '_mean': val for key, val in eval_dict.items()}
                            runtime_dict = {str(key) + '_mean': val for key, val in runtime_dict.items()}
                            new_row = {'model': current_model}
                            new_row.update(eval_dict)
                            new_row.update(runtime_dict)
                            if 'nested' in pattern:
                                eval_dict_std = {str(key) + '_std': val for key, val in eval_dict_std.items()}
                                runtime_dict_std = {str(key) + '_std': val for key, val in runtime_dict_std.items()}
                                new_row.update(eval_dict_std)
                                new_row.update(runtime_dict_std)
                            if overview_df is None:
                                overview_df = pd.DataFrame(new_row, index=[0])
                            else:
                                overview_df = pd.concat([overview_df, pd.DataFrame(new_row, index=[0])],
                                                        ignore_index=True)
                        except Exception as exc:
                            print('No Results File')
                            continue
                        if 'nested' in pattern:
                            for outerfold_path in path.glob('outerfold*'):
                                runtime_file = pd.read_csv(
                                    outerfold_path.joinpath(current_model, current_model + '_runtime_overview.csv')
                                )
                                runtime_file.to_excel(
                                    writer, sheet_name=current_model + '_of' + outerfold_path.parts[-1].split('_')[-1]
                                                       + '_runtime',
                                    index=False
                                )
                        else:
                            runtime_file = \
                                pd.read_csv(path.joinpath(current_model, current_model + '_runtime_overview.csv'))
                            runtime_file.to_excel(writer, sheet_name=current_model + '_runtime', index=False)
                if overview_df is None:
                    continue
                overview_df.to_excel(writer, sheet_name='Overview_results', index=False)
                overview_df.to_csv(phenotype_folder.joinpath('Results_summary_' + phenotype + '_' + pattern + '.csv'))
                writer.sheets['Overview_results'].activate()
                writer.save()
    for pattern in datasplit_maf_patterns:
        overview_sheet = pd.DataFrame(
            columns=helper_functions.get_list_of_implemented_models()
        )
        writer = pd.ExcelWriter(
            results_directory_genotype_level.joinpath('Results_summary_all_phenotypes_' + pattern + '.xlsx'),
            engine='xlsxwriter'
        )
        paths = [path for path in list(results_directory_genotype_level.rglob('Results_summary*' + pattern + '*.csv'))
                 if 'all_phenotypes' not in str(path)]
        overview_sheet['phenotype'] = [path.parts[-2] for path in paths]
        overview_sheet.set_index('phenotype', drop=True, inplace=True)
        for results_summary_path in paths:
            results_summary = pd.read_csv(results_summary_path)
            results_summary.to_excel(
                writer, sheet_name=results_summary_path.parts[-2],
                index=False
            )
            phenotype = results_summary_path.parts[-2]
            eval_metric = 'test_explained_variance' \
                if any(['test_explained_variance' in col for col in results_summary.columns]) else 'test_f1_score'
            if 'nested' in pattern:
                for row in results_summary.iterrows():
                    overview_sheet.at[phenotype, row[1]['model']] = "{:.3f} +- {:.3f}".format(
                        row[1][eval_metric + '_mean'], row[1][eval_metric + '_std'])
            else:
                for row in results_summary.iterrows():
                    overview_sheet.at[phenotype, row[1]['model']] = "{:.3f}".format(row[1][eval_metric + '_mean'])
        overview_sheet.dropna(axis=1, inplace=True, how='all')  # drop model column if all results are missing
        overview_sheet.to_excel(writer, sheet_name='Overview')
        overview_sheet.to_csv(
            results_directory_genotype_level.joinpath('Results_summary_all_phenotypes_' + pattern + '.csv')
        )
        writer.sheets['Overview'].activate()
        writer.save()


def result_string_to_dictionary(result_string: str) -> dict:
    """
    Convert result string saved in a .csv file to a dictionary

    :param result_string: string from .csv file

    :return: dictionary with info from .csv file
    """
    key_value_strings = result_string.split('\\')[0][2:-2].replace('\'', '').split(',')
    dict_result = {}
    for key_value_string in key_value_strings:
        key = key_value_string.split(':')[0].strip()
        value = key_value_string.split(':')[1].strip()
        try:
            value = float(value)
            value = int(value) if value == int(value) else value
        except:
            value = value
        dict_result[key] = value
    return dict_result


def plot_heatmap_results(path_to_results_summary_csv: str, save_dir: str):
    """
    Generate a heatmap based on the results summary .csv file

    :param path_to_results_summary_csv: path to the results summary .csv file
    :param save_dir: directory to save the plots
    """
    path_to_results_summary_csv = pathlib.Path(path_to_results_summary_csv)
    save_dir = pathlib.Path(save_dir) if save_dir is not None else path_to_results_summary_csv.parents[0]
    fig, ax = plt.subplots(figsize=(12, 6))

    results_overview = pd.read_csv(path_to_results_summary_csv)
    results_overview.set_index("phenotype", inplace=True)
    models = [mod for mod in results_overview.columns if mod != 'phenotype']

    if 'nested' in path_to_results_summary_csv.parts[-1]:
        types = ['mean', 'std']
    else:
        types = ['mean']
    plot_data_full = pd.DataFrame(columns=[mod + '_' + type for mod in models for type in types] + ['phenotype'])
    plot_data_full['phenotype'] = results_overview.index
    plot_data_full.set_index("phenotype", inplace=True)
    for row in results_overview.iterrows():
        for model in models:
            result = row[1][model]
            if 'std' in types:
                plot_data_full.at[row[0], model + '_std'] = float(result.split('+-')[1])
                plot_data_full.at[row[0], model + '_mean'] = float(result.split('+-')[0])
            else:
                plot_data_full.at[row[0], model + '_mean'] = float(result)
    plot_data_mean = plot_data_full.filter(regex='mean').astype(float)
    row_max = plot_data_mean.idxmax(axis=1)
    sns.heatmap(data=plot_data_mean, cmap="Spectral", cbar_kws={"shrink": .75},
                annot=results_overview, fmt='', linewidths=1.5, linecolor='white', cbar=True, annot_kws={"size": 12})
    ax.set_xticklabels(results_overview.columns, rotation=0)
    ax.set_yticklabels(plot_data_full.index, rotation=0)
    ax.tick_params(top=False,
                   bottom=False,
                   left=False,
                   right=False,
                   labelleft=True,
                   labelbottom=True)
    for row, index in enumerate(plot_data_mean.index):
        position = results_overview.columns.get_loc(row_max[index].split('_')[0])
        ax.add_patch(Rectangle((position, row), 1, 1, fill=False, edgecolor='0', lw=1.5))

    fig.tight_layout()
    plt.savefig(save_dir.joinpath('heatmap_' + path_to_results_summary_csv.parts[-1].split('.')[0] + '.pdf'),
                bbox_inches='tight', dpi=600)
