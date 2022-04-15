import glob
import pandas as pd
import os


def summarize_results_per_phenotype_and_datasplit(results_directory_genotype_level: str):
    """
    Summarize the results for each phenotype and datasplit for all models and save in a file.

    :param results_directory_genotype_level: Results directory at the level of the name of the genotype matrix
    """
    for study in os.listdir(results_directory_genotype_level):
        for phenotype in os.listdir(results_directory_genotype_level + '/' + study):
            print('++++++++++++++ PHENOTYPE ' + phenotype + ' ++++++++++++++')
            current_directory = results_directory_genotype_level + '/' + study + '/' + phenotype + '/'
            subdirs = \
                [name for name in os.listdir(current_directory) if os.path.isdir(os.path.join(current_directory, name))]
            datasplit_maf_patterns = \
                set(['_'.join(path.split('/')[-1].split('_')[:3]) for path in subdirs])
            for pattern in datasplit_maf_patterns:
                writer = pd.ExcelWriter(
                    current_directory + '/Detailed_results_summary_' + phenotype + '_' + pattern + '.xlsx',
                    engine='xlsxwriter'
                )
                overview_df = None
                print('----- Datasplit pattern ' + pattern + ' -----')
                for path in glob.glob(current_directory + pattern + '*'):
                    models = path.split('/')[-1].split('_')[3:-2]
                    for current_model in models:
                        print('### Results for ' + current_model + ' ###')
                        try:
                            results_file = glob.glob(path + '/*.csv')[0]
                            results = pd.read_csv(results_file)
                            results = results.loc[:, [current_model in col for col in results.columns]]
                            idx = 0
                            if 'nested-cv' in pattern:
                                idx = -2
                                eval_dict_std = result_string_to_dictionary(
                                    result_string=results[current_model + '___eval_metrics'][-1])
                                runtime_dict_std = result_string_to_dictionary(
                                    result_string=results[current_model + '___runtime_metrics'][-1]
                                )
                            eval_dict = result_string_to_dictionary(
                                result_string=results[current_model + '___eval_metrics'][idx]
                            )
                            runtime_dict = result_string_to_dictionary(
                                result_string=results[current_model + '___runtime_metrics'][idx]
                            )
                            results.to_excel(writer, sheet_name=current_model + '_results', index=False)
                            eval_dict = {str(key) + '_mean': val for key, val in eval_dict.items()}
                            runtime_dict = {str(key) + '_mean': val for key, val in runtime_dict.items()}
                            new_row = {'model': current_model}
                            new_row.update(eval_dict)
                            new_row.update(runtime_dict)
                            if 'nested-cv' in pattern:
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
                            print(exc)
                            print('No Results File')
                        runtime_file = \
                            pd.read_csv(path + '/' + current_model + '/' + current_model + '_runtime_overview.csv')
                        runtime_file.to_excel(writer, sheet_name=current_model + '_runtime', index=False)
                overview_df.to_excel(writer, sheet_name='Overview_results', index=False)
                overview_df.to_csv(current_directory + '/Results_summary_' + phenotype + '_' + pattern + '.csv')
            writer.sheets['Overview_results'].activate()
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
