import numpy as np
import datetime
import argparse

import pandas as pd
import pathlib

from ..preprocess import base_dataset, encoding_functions, raw_data_functions
from ..utils import helper_functions, check_functions
from ..model import _model_functions
from ..evaluation import eval_metrics
from . import generate_feat_impo


def retrain_on_new_data(path_to_model_results_folder: str,
                        data_dir: str, genotype_matrix: str, phenotype_matrix: str, phenotype: str,
                        encoding: str = None, maf_percentage: int = 0, save_dir: str = None,
                        datasplit: str = 'nested-cv', n_outerfolds: int = 5, n_innerfolds: int = 5,
                        test_set_size_percentage: int = 20, val_set_size_percentage: int = 20,
                        save_final_model: bool = False,
                        batch_size: int = 32, n_epochs: int = 100000):

    # create Path
    path_to_model_results_folder = pathlib.Path(path_to_model_results_folder)
    if not check_functions.check_exist_directories(list_of_dirs=[path_to_model_results_folder]):
        raise Exception("See output above. Problems with specified directories")
    model_name = path_to_model_results_folder.parts[-1]
    models = [model_name]
    data_dir = pathlib.Path(data_dir)
    # set save directory
    save_dir = data_dir if save_dir is None else pathlib.Path(save_dir)
    save_dir = save_dir if save_dir.is_absolute() else save_dir.resolve()
    # Check all arguments
    check_functions.check_all_specified_arguments(arguments=locals())
    # prepare all data files
    raw_data_functions.prepare_data_files(
        data_dir=data_dir, genotype_matrix_name=genotype_matrix, phenotype_matrix_name=phenotype_matrix,
        phenotype=phenotype, datasplit=datasplit, n_outerfolds=n_outerfolds, n_innerfolds=n_innerfolds,
        test_set_size_percentage=test_set_size_percentage, val_set_size_percentage=val_set_size_percentage,
        models=models, user_encoding=encoding, maf_percentage=maf_percentage
    )
    encoding = helper_functions.get_mapping_name_to_class()[model_name].standard_encoding \
        if encoding is None else encoding
    print("Load new dataset")
    new_dataset = base_dataset.Dataset(
        data_dir=data_dir, genotype_matrix_name=genotype_matrix, phenotype_matrix_name=phenotype_matrix,
        phenotype=phenotype, datasplit=datasplit, n_outerfolds=n_outerfolds, n_innerfolds=n_innerfolds,
        test_set_size_percentage=test_set_size_percentage, val_set_size_percentage=val_set_size_percentage,
        encoding=encoding, maf_percentage=maf_percentage
    )
    # train on new data
    if datasplit == 'train-val-test':
        datasplit_params = [val_set_size_percentage, test_set_size_percentage]
    elif datasplit == 'cv-test':
        datasplit_params = [n_innerfolds, test_set_size_percentage]
    elif datasplit == 'nested-cv':
        datasplit_params = [n_outerfolds, n_innerfolds]
    datasplit_subpath = helper_functions.get_subpath_for_datasplit(
        datasplit=datasplit, datasplit_params=datasplit_params
    )
    models_start_time = '+'.join(models) + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_path = save_dir.joinpath('results', genotype_matrix.split('.')[0], phenotype_matrix.split('.')[0],
                                  phenotype, datasplit + '_' + datasplit_subpath + '_MAF' + str(maf_percentage) +
                                  '_' + models_start_time)
    helper_functions.set_all_seeds()
    for outerfold_number in range(n_outerfolds):
        print("train on new dataset")
        if 'nested' in datasplit:
            print("outerfold " + str(outerfold_number))
        model = _model_functions.retrain_model_with_results_file(
            results_file_path=path_to_model_results_folder, model_name=model_name, datasplit=datasplit,
            outerfold_number=outerfold_number, dataset=new_dataset
        )
        if save_final_model:
            save_path = base_path.joinpath('outerfold_' + str(outerfold_number), model_name) \
                if 'nested' in datasplit else base_path.joinpath(model_name)
            if not save_path.exists():
                save_path.mkdir(parents=True)
            model.save_model(path=save_path, filename='final_retrained_model')
        outerfold_info = new_dataset.datasplit_indices['outerfold_' + str(outerfold_number)]
        X_test, y_test, sample_ids_test = \
            new_dataset.X_full[outerfold_info['test']], new_dataset.y_full[outerfold_info['test']], \
            new_dataset.sample_ids_full[outerfold_info['test']]
        X_retrain, y_retrain, sample_ids_retrain = \
            new_dataset.X_full[~np.isin(np.arange(len(new_dataset.X_full)), outerfold_info['test'])], \
            new_dataset.y_full[~np.isin(np.arange(len(new_dataset.y_full)), outerfold_info['test'])], \
            new_dataset.sample_ids_full[~np.isin(np.arange(len(new_dataset.sample_ids_full)), outerfold_info['test'])]
        y_pred_retrain = model.predict(X_in=X_retrain)
        y_pred_test = model.predict(X_in=X_test)

        # Evaluate and save results
        eval_scores = \
            eval_metrics.get_evaluation_report(y_pred=y_pred_test, y_true=y_test, task=model.task, prefix='test_')
        final_results = pd.DataFrame(index=range(0, new_dataset.y_full.shape[0]))
        final_results.at[0:len(sample_ids_retrain) - 1, 'sample_ids_retrain'] = sample_ids_retrain.flatten()
        final_results.at[0:len(y_pred_retrain) - 1, 'y_pred_retrain'] = y_pred_retrain.flatten()
        final_results.at[0:len(y_retrain) - 1, 'y_true_retrain'] = y_retrain.flatten()
        final_results.at[0:len(sample_ids_test) - 1, 'sample_ids_test'] = sample_ids_test.flatten()
        final_results.at[0:len(y_pred_test) - 1, 'y_pred_test'] = y_pred_test.flatten()
        final_results.at[0:len(y_test) - 1, 'y_true_test'] = y_test.flatten()

        for metric, value in eval_scores.items():
            final_results.at[0, metric] = value
        final_results.at[0, 'base_model_path'] = path_to_model_results_folder
        save_path = base_path.joinpath('outerfold_' + str(outerfold_number), model_name) \
            if 'nested' in datasplit else base_path.joinpath(model_name)
        if not save_path.exists():
            save_path.mkdir(parents=True)
        final_results.to_csv(save_path.joinpath('final_model_test_results.csv'),
                             sep=',', decimal='.', float_format='%.10f', index=False)
    # generate feature importances
    generate_feat_impo.post_generate_feature_importances(
        data_dir=str(data_dir),
        results_directory_genotype_level=str(save_dir.joinpath('results', genotype_matrix.split('.')[0]))
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dd", "--data_dir", type=str,
                        help="Provide the full path of the data directory that contains the geno- and phenotype "
                             "files you want to optimize on")
    parser.add_argument("-gm", "--genotype_matrix", type=str,
                        help="Provide the name of the genotype matrix you want to predict on")
    parser.add_argument("-pm", "--phenotype_matrix", type=str,
                        help="Provide the name of the phenotype matrix you want to predict on")
    parser.add_argument("-p", "--phenotype", type=str,
                        help="Provide the name of the phenotype you want to predict on")
    parser.add_argument("-sd", "--save_dir", type=str,
                        help="Define the save directory for the results.")
    parser.add_argument("-rd", "--results_dir_model", type=str,
                        help="Provide the full path of the directory where your results of the model "
                             "you want to use are stored")

    parser.add_argument("-enc", "--encoding", type=str, default=None,
                        help="specify the encoding to use. Caution: has to be a possible encoding for the model to use."
                             "Valid arguments are: " + str(encoding_functions.get_list_of_encodings()))
    # Preprocess Params #
    parser.add_argument("-maf", "--maf_percentage", type=int, default=0,
                        help="specify the minor allele frequency (as percentage value). "
                             "specify 0 if you do not want a maf filter.")
    parser.add_argument("-split", "--datasplit", type=str, default='nested-cv',
                        help="specify the data split to use: 'nested-cv' | 'cv-test' | 'train-val-test'"
                             "Default values are 5 folds, train-test-split to 80/20 and train-val-test to 60/20/20")
    parser.add_argument("-testperc", "--test_set_size_percentage", type=int, default=20,
                        help="specify the size of the test set in percentage. "
                             "Standard is 20, only relevant for 'cv-test' and 'train-val-test'")
    parser.add_argument("-valperc", "--val_set_size_percentage", type=int, default=20,
                        help="specify the size of the validation set in percentage. "
                             "Standard is 20, only relevant for 'train-val-test'")
    parser.add_argument("-of", "--n_outerfolds", type=int, default=3,
                        help="specify the number of outerfolds to use for 'nested_cv'"
                             "Standard is 5, only relevant for 'nested_cv'")
    parser.add_argument("-folds", "--n_innerfolds", type=int, default=5,
                        help="specify the number of innerfolds/folds to use for 'nested_cv' respectively 'cv-test'"
                             "Standard is 5, only relevant for 'nested_cv' and 'cv-test'")

    # Model and Optimization Params #
    parser.add_argument("-sf", "--save_final_model", type=bool, default=False,
                        help="save the final model to hard drive "
                             "(caution: some models may use a lot of disk space, "
                             "unfitted models that can be retrained are already saved by default)")

    # Only relevant for Neural Networks #
    parser.add_argument("-bs", "--batch_size", type=int, default=None,
                        help="Only relevant for neural networks: define the batch size.")
    parser.add_argument("-ep", "--n_epochs", type=int, default=None,
                        help="Only relevant for neural networks: define the number of epochs. If nothing is specified,"
                             "it will be considered as a hyperparameter for optimization")

    args = vars(parser.parse_args())

    retrain_on_new_data(**args)
