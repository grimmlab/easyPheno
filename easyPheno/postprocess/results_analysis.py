import glob
import pandas as pd
import os


def summarize_results_per_phenotype_and_datasplit(results_directory_genotype_level: str):
    """
    Summarize the results for each phenotype and datasplit for all models and save in a file.

    :param results_directory_genotype_level: Results directory at the level of the name of the genotype matrix
    """
    for study in os.listdir(results_directory_genotype_level):
        if not os.path.isdir(results_directory_genotype_level + '/' + study):
            continue
        for phenotype in os.listdir(results_directory_genotype_level + '/' + study):
            current_directory = results_directory_genotype_level + '/' + study + '/' + phenotype + '/'
            if not os.path.isdir(current_directory):
                continue
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
                    models = path.split('/')[-1].split('_')[3].split('+')
                    for current_model in models:
                        print('### Results for ' + current_model + ' ###')
                        try:
                            results_file = glob.glob(path + '/*.csv')[0]
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
                            print(exc)
                            print('No Results File')
                            continue
                        if 'nested' in pattern:
                            for outerfold_path in glob.glob(path + '/outerfold*'):
                                runtime_file = pd.read_csv(
                                    outerfold_path + '/' + current_model + '/' + current_model + '_runtime_overview.csv'
                                )
                                runtime_file.to_excel(
                                    writer, sheet_name=current_model + '_of' + outerfold_path.split('_')[-1] \
                                    + '_runtime',
                                    index=False
                                )
                        else:
                            runtime_file = \
                                pd.read_csv(path + '/' + current_model + '/' + current_model + '_runtime_overview.csv')
                            runtime_file.to_excel(writer, sheet_name=current_model + '_runtime', index=False)
                overview_df.to_excel(writer, sheet_name='Overview_results', index=False)
                overview_df.to_csv(current_directory + '/Results_summary_' + phenotype + '_' + pattern + '.csv')
                writer.sheets['Overview_results'].activate()
                writer.save()
    overview_sheet = pd.DataFrame(
        columns=['xgboost', 'randomforest', 'linearregression', 'svm', 'mlp', 'cnn', 'localcnn', 'blup']
    )
    for pattern in datasplit_maf_patterns:
        writer = pd.ExcelWriter(
            results_directory_genotype_level + '/Results_summary_' + pattern + '.xlsx',
            engine='xlsxwriter'
        )
        if 'Simulation' in results_directory_genotype_level:
            paths = sorted(glob.glob(results_directory_genotype_level + '/*/*/Results_summary*' + pattern + '*.csv'),
                           key=lambda x: int(x.split('/')[-2].split('_')[0][3:]))
        else:
            paths = glob.glob(results_directory_genotype_level + '/*/*/Results_summary*' + pattern + '*.csv')
        overview_sheet['exp'] = [path.split('/')[-1].split('_')[2] for path in paths]
        overview_sheet.set_index('exp', drop=True, inplace=True)
        for results_summary_path in paths:
            results_summary = pd.read_csv(results_summary_path)
            results_summary.to_excel(
                writer, sheet_name=results_summary_path.split('/')[-2],
                index=False
            )
            exp = results_summary_path.split('/')[-1].split('_')[2]
            eval_metric = 'test_explained_variance' \
                if any(['test_explained_variance' in col for col in results_summary.columns]) else 'test_f1_score'
            if eval_metric + '_std' in results_summary.columns:
                for row in results_summary.iterrows():
                    overview_sheet.at[exp, row[1]['model']] = "{:.3f} +- {:.3f}".format(
                        row[1][eval_metric + '_mean'], row[1][eval_metric + '_std'])
            else:
                for row in results_summary.iterrows():
                    overview_sheet.at[exp, row[1]['model']] = "{:.3f}".format(row[1][eval_metric + '_mean'])
    overview_sheet.to_excel(writer, sheet_name='Overview')
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


#summarize_results_per_phenotype_and_datasplit(results_directory_genotype_level=...
#)
