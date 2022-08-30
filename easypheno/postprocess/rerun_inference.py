import argparse
import numpy as np
import pandas as pd
import pathlib
import shutil

from ..model import _model_functions
from ..utils import check_functions, helper_functions
from ..preprocess import base_dataset
from ..evaluation import eval_metrics


if __name__ == "__main__":
    """
    Run to apply the specified model on a dataset containing new samples.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-dd", "--data_dir", type=str, default=None,
                        help="Provide the full path of the old data directory (that contains the geno- and phenotype "
                             "files as well as the index file the model was trained on). "
                             "Only needed if the final model was not saved.")
    parser.add_argument("-gm", "--genotype_matrix", type=str,
                        help="Provide the name of the new genotype matrix you want to predict on")
    parser.add_argument("-pm", "--phenotype_matrix", type=str,
                        help="Provide the name of the new phenotype matrix you want to predict on")
    parser.add_argument("-sd", "--save_dir", type=str,
                        help="Define the save directory for the results.")
    parser.add_argument("-rd", "--results_dir_model", type=str,
                        help="Provide the full path of the directory where your results of the model "
                             "you want to use are stored")
    args = vars(parser.parse_args())
    data_dir = args['data_dir']
    genotype_matrix = args['genotype_matrix']
    phenotype_matrix = args['phenotype_matrix']
    save_dir = args["save_dir"]
    results_directory_model = args['results_dir_model']
    str_results_dir_model = args['results_dir_model']
    overall_results = {}
    for fold in range(3):
        helper_functions.set_all_seeds(42)
        results_directory_model = \
            pathlib.Path(str_results_dir_model[:str_results_dir_model.rindex('outerfold')] +
                         'outerfold_' + str(fold) +
                         str_results_dir_model[str_results_dir_model.rindex('outerfold') + 11:])

        data_dir = pathlib.Path(data_dir)
        save_dir = pathlib.Path(save_dir)
        new_results_dir = save_dir.joinpath(*results_directory_model.parts[-6:])
        check_functions.check_exist_directories(list_of_dirs=[new_results_dir], create_if_not_exist=True)
        for file in helper_functions.get_all_files(results_directory_model):
            shutil.copyfile(file, new_results_dir.joinpath(file.name))
        shutil.move(new_results_dir.joinpath('final_model_test_results.csv'),
                    new_results_dir.joinpath('final_model_test_results_old.csv'))
        # Check user inputs
        print("Checking user inputs")
        if not check_functions.check_exist_directories(
                list_of_dirs=[results_directory_model, data_dir, save_dir]):
            raise Exception("See output above. Problems with specified directories")

        result_folder_name = results_directory_model.parts[-3] if 'nested' in str(results_directory_model) \
            else results_directory_model.parts[-2]
        model_name = results_directory_model.parts[-1]
        if 'fromR' in model_name:
            import easypheno.model
            easypheno.model.__all__.extend(['_bayesfromR', 'bayesAfromR', 'bayesBfromR', 'bayesCfromR'])
        datasplit, n_outerfolds, n_innerfolds, val_set_size_percentage, test_set_size_percentage, maf_perc = \
            helper_functions.get_datasplit_config_info_for_resultfolder(resultfolder=result_folder_name)
        nested_offset = -1 if 'nested' in datasplit else 0
        phenotype = results_directory_model.parts[-3 + nested_offset]
        if not check_functions.check_exist_files(
                list_of_files=[
                    data_dir.joinpath(phenotype_matrix), data_dir.joinpath(genotype_matrix),
                    data_dir.joinpath(base_dataset.Dataset.get_index_file_name(
                        genotype_matrix_name=genotype_matrix, phenotype_matrix_name=phenotype_matrix,
                        phenotype=phenotype))]):
            raise Exception("See output above. Problems with specified files.")


        encoding = helper_functions.get_mapping_name_to_class()[model_name].standard_encoding

        # Prepare the model
        outerfold_number = int(results_directory_model.parent.parts[-1].split('_')[1]) if 'nested' in datasplit else 0
        models = results_directory_model.parts[-2 + nested_offset].split('_')[3].split('+')
        results_file_path = \
            results_directory_model.parents[0 - nested_offset].joinpath('Results_overview_' + '_'.join(models) + '.csv')
        print("Retraining model")
        print("Loading old dataset")
        dataset = base_dataset.Dataset(
            data_dir=data_dir, genotype_matrix_name=genotype_matrix,
            phenotype_matrix_name=phenotype_matrix,
            phenotype=phenotype, datasplit=datasplit, n_outerfolds=n_outerfolds, n_innerfolds=n_innerfolds,
            test_set_size_percentage=test_set_size_percentage, val_set_size_percentage=val_set_size_percentage,
            encoding=encoding, maf_percentage=maf_perc
        )
        if not results_file_path.is_file():
            raise Exception("Results Overview file not existing. Please check: " + str(results_file_path))
        model = _model_functions.retrain_model_with_results_file(
            results_file_path=results_file_path, model_name=model_name, datasplit=datasplit,
            outerfold_number=outerfold_number, dataset=dataset
        )

        # Do inference and save results
        print('-----------------------------------------------')
        print("Inference on new data for " + model_name)
        outerfold_info = dataset.datasplit_indices['outerfold_' + str(outerfold_number)]
        X_test, y_test, sample_ids_test = \
            dataset.X_full[outerfold_info['test']], dataset.y_full[outerfold_info['test']], \
            dataset.sample_ids_full[outerfold_info['test']]
        X_retrain, y_retrain, sample_ids_retrain = \
            dataset.X_full[~np.isin(np.arange(len(dataset.X_full)), outerfold_info['test'])], \
            dataset.y_full[~np.isin(np.arange(len(dataset.y_full)), outerfold_info['test'])], \
            dataset.sample_ids_full[~np.isin(np.arange(len(dataset.sample_ids_full)),
                                                  outerfold_info['test'])]
        y_pred_retrain = model.predict(X_in=X_retrain)
        y_pred_test = model.predict(X_in=X_test)
        eval_scores = \
            eval_metrics.get_evaluation_report(y_pred=y_pred_test, y_true=y_test, task=model.task,
                                               prefix='test_')
        print('## Results on test set ##')
        print(eval_scores)
        final_results = pd.DataFrame(index=range(0, dataset.y_full.shape[0]))
        final_results.at[0:len(sample_ids_retrain) - 1, 'sample_ids_retrain'] = sample_ids_retrain.flatten()
        final_results.at[0:len(y_pred_retrain) - 1, 'y_pred_retrain'] = y_pred_retrain.flatten()
        final_results.at[0:len(y_retrain) - 1, 'y_true_retrain'] = y_retrain.flatten()
        final_results.at[0:len(sample_ids_test) - 1, 'sample_ids_test'] = sample_ids_test.flatten()
        final_results.at[0:len(y_pred_test) - 1, 'y_pred_test'] = y_pred_test.flatten()
        final_results.at[0:len(y_test) - 1, 'y_true_test'] = y_test.flatten()
        for metric, value in eval_scores.items():
            final_results.at[0, metric] = value
        final_results.to_csv(new_results_dir.joinpath(
            'final_model_test_results.csv'),
            sep=',', decimal='.', float_format='%.10f', index=False
        )
        key = 'outerfold_' + str(fold)
        best_params = model.optuna_trial.params
        overall_results[key] = {'best_params': best_params, 'eval_metrics': eval_scores,
                                'runtime_metrics': np.nan}
    model_overview = {model_name: overall_results}
    path_overview_file = new_results_dir.parent.parent.joinpath(
        'Results_overview_' + '_'.join([model_name]) + '.csv')
