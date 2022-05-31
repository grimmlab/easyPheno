import pandas as pd
import numpy as np
import os
import re
import time
import csv

from ..preprocess import base_dataset
from ..utils import helper_functions
from ..evaluation import eval_metrics
from ..model import _param_free_base_model


class ParamFreeFitting:
    """
    Class that contains all info for the whole optimization using optuna for one model and dataset.

    **Attributes**

        - task (*str*): ML task (regression or classification) depending on target variable
        - current_model_name (*str*): name of the current model according to naming of .py file in package model
        - dataset (:obj:`~easyPheno.preprocess.base_dataset.Dataset`): dataset to use for optimization run
        - datasplit_subpath (*str*): subpath with datasplit info relevant for saving / naming
        - base_path (*str*): base_path for save_path
        - save_path (*str*): path for model and results storing
        - user_input_params (*dict*): all params handed over to the constructor that are needed in the whole class

    :param save_dir: directory for saving the results.
    :param genotype_matrix_name: name of the genotype matrix including datatype ending
    :param phenotype_matrix_name: name of the phenotype matrix including datatype ending
    :param phenotype: name of the phenotype to predict
    :param n_outerfolds: number of outerfolds relevant for nested-cv
    :param n_innerfolds: number of folds relevant for nested-cv and cv-test
    :param test_set_size_percentage: size of the test set relevant for cv-test and train-val-test
    :param val_set_size_percentage: size of the validation set relevant for train-val-test
    :param maf_percentage: threshold for MAF filter as percentage value
    :param save_final_model: specify if the final model should be saved
    :param task: ML task (regression or classification) depending on target variable
    :param current_model_name: name of the current model according to naming of .py file in package model
    :param dataset: dataset to use for optimization run
    :param models_start_time: optimized models and starting time of the optimization run for saving purposes
    """

    def __init__(self, save_dir: str, genotype_matrix_name: str, phenotype_matrix_name: str, phenotype: str,
                 n_outerfolds: int, n_innerfolds: int, val_set_size_percentage: int, test_set_size_percentage: int,
                 maf_percentage: int, save_final_model: bool, task: str, current_model_name: str,
                 dataset: base_dataset.Dataset, models_start_time: str):
        self.current_model_name = current_model_name
        self.task = task
        self.dataset = dataset
        if self.dataset.datasplit == 'train-val-test':
            datasplit_params = [val_set_size_percentage, test_set_size_percentage]
        elif self.dataset.datasplit == 'cv-test':
            datasplit_params = [n_innerfolds, test_set_size_percentage]
        elif self.dataset.datasplit == 'nested-cv':
            datasplit_params = [n_outerfolds, n_innerfolds]
        self.datasplit_subpath = helper_functions.get_subpath_for_datasplit(
            datasplit=self.dataset.datasplit, datasplit_params=datasplit_params
        )
        self.base_path = save_dir + '/results/' + genotype_matrix_name.split('.')[0] + '/' + \
                         phenotype_matrix_name.split('.')[0] + '/' + phenotype + '/' + self.dataset.datasplit + '_' + \
                         self.datasplit_subpath + '_MAF' + str(maf_percentage) + '_' + models_start_time + '/' \
                         + current_model_name + '/'
        self.save_path = self.base_path
        self.user_input_params = locals()  # distribute all handed over params in whole class

    def run_fitting(self):
        """
        Run fitting of parameter-free models

        :return: dictionary with results overview
        """
        # Iterate over outerfolds
        # (according to structure described in base_dataset.Dataset, only for nested-cv multiple outerfolds exist)
        helper_functions.set_all_seeds()
        overall_results = {}
        for outerfold_name, outerfold_info in self.dataset.datasplit_indices.items():
            if self.dataset.datasplit == 'nested-cv':
                # Only print outerfold info for nested-cv as it does not apply for the other splits
                print("## Starting Fitting for " + outerfold_name + " ##")
                end_ind = [m.end(0) for m in re.finditer(pattern='/', string=self.base_path)][-2]
                self.save_path = self.base_path[:end_ind] + outerfold_name + '/' + self.base_path[end_ind:]
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

            # get datasets
            X_test, y_test, sample_ids_test = \
                self.dataset.X_full[outerfold_info['test']], self.dataset.y_full[outerfold_info['test']], \
                self.dataset.sample_ids_full[outerfold_info['test']]
            X_train, y_train, sample_ids_train = \
                self.dataset.X_full[~np.isin(np.arange(len(self.dataset.X_full)), outerfold_info['test'])], \
                self.dataset.y_full[~np.isin(np.arange(len(self.dataset.y_full)), outerfold_info['test'])], \
                self.dataset.sample_ids_full[~np.isin(np.arange(len(self.dataset.sample_ids_full)),
                                                      outerfold_info['test'])]
            # create and fit model
            model: _param_free_base_model.ParamFreeBaseModel = \
                helper_functions.get_mapping_name_to_class()[self.current_model_name](
                    task=self.task,
                )
            model.save_model(path=self.save_path, filename='unfitted_model')
            helper_functions.set_all_seeds()
            start_process_time = time.process_time()
            start_realclock_time = time.time()
            y_pred_train = model.fit(X=X_train, y=y_train)
            process_time_s = time.process_time() - start_process_time
            real_time_s = time.time() - start_realclock_time
            self.write_runtime_csv(dict_runtime={
                'Trial': 'train', 'process_time_s': process_time_s, 'real_time_s': real_time_s,
                'params': None, 'note': None}
            )
            if self.user_input_params["save_final_model"]:
                model.save_model(path=self.save_path, filename='final_retrained_model')
            y_pred_test = model.predict(X_in=X_test)

            feat_import_df = None
            if self.current_model_name in ['blup']:
                feat_import_df = self.get_feature_importance(model=model, X=X_test, y=y_test)
            # Evaluate and save results
            eval_scores = \
                eval_metrics.get_evaluation_report(y_pred=y_pred_test, y_true=y_test, task=self.task, prefix='test_')
            print('## Results on test set ##')
            print(eval_scores)
            final_results = pd.DataFrame(index=range(0, self.dataset.y_full.shape[0]))
            final_results.at[0:len(sample_ids_train) - 1, 'sample_ids_train'] = sample_ids_train.flatten()
            final_results.at[0:len(y_pred_train) - 1, 'y_pred_train'] = y_pred_train.flatten()
            final_results.at[0:len(y_train) - 1, 'y_true_train'] = y_train.flatten()
            final_results.at[0:len(sample_ids_test) - 1, 'sample_ids_test'] = sample_ids_test.flatten()
            final_results.at[0:len(y_pred_test) - 1, 'y_pred_test'] = y_pred_test.flatten()
            final_results.at[0:len(y_test) - 1, 'y_true_test'] = y_test.flatten()
            for metric, value in eval_scores.items():
                final_results.at[0, metric] = value
            final_results.to_csv(
                self.save_path + 'final_model_test_results.csv', sep=',', decimal='.', float_format='%.10f', index=False
            )

            runtime_metrics = {
                'process_time_mean': process_time_s, 'process_time_std': 0,
                'process_time_max': process_time_s, 'process_time_min': process_time_s,
                'real_time_mean': real_time_s, 'real_time_std': 0,
                'real_time_max': real_time_s, 'real_time_min': real_time_s
            }
            key = outerfold_name if self.dataset.datasplit == 'nested-cv' else 'Test'
            overall_results[key] = {'best_params': None, 'eval_metrics': eval_scores,
                                    'runtime_metrics': runtime_metrics}

            if feat_import_df is not None:
                feat_import_df.to_csv(
                    self.save_path + 'final_model_feature_importances.csv', sep=',', decimal='.', float_format='%.10f',
                    index=False
                )
        return overall_results

    def write_runtime_csv(self, dict_runtime: dict):
        """
        Write runtime info to runtime csv file

        :param dict_runtime: dictionary with runtime information
        """
        with open(self.save_path + self.current_model_name + '_runtime_overview.csv', 'a') as runtime_file:
            headers = ['Trial', 'process_time_s', 'real_time_s', 'params', 'note']
            writer = csv.DictWriter(f=runtime_file, fieldnames=headers)
            if runtime_file.tell() == 0:
                writer.writeheader()
            writer.writerow(dict_runtime)

    def get_feature_importance(self, model: _param_free_base_model.ParamFreeBaseModel, X: np.array, y: np.array,
                               top_n: int = 1000) -> pd.DataFrame:
        """
        Get feature importances for models that possess such a feature, e.g. BLUP

        :param model: model to analyze
        :param X: feature matrix for permutation
        :param y: target vector for permutation
        :param top_n: top n features to select

        :return: DataFrame with feature importance information
        """

        top_n = min(len(self.dataset.snp_ids), top_n)
        feat_import_df = pd.DataFrame()
        if self.current_model_name in ['blup']:
            feat_imps = model.u
            dims = feat_imps.shape[0] if len(feat_imps.shape) > 1 else 1
            for dim in range(dims):
                coef = feat_imps[dim] if len(feat_imps.shape) > 1 else feat_imps
                sorted_idx = coef.argsort()[::-1][:top_n]
                feat_import_df['snp_ids_' + str(dim)] = self.dataset.snp_ids[sorted_idx]
                feat_import_df['coefficients_' + str(dim)] = coef[sorted_idx]

        return feat_import_df