import argparse
import datetime
import optuna
import pandas as pd
import sklearn
import numpy as np
import os
import glob
import shutil
import re
import gc
import time
import csv

import torch.cuda
import tensorflow as tf

import utils
from preprocess import base_dataset
from utils import helper_functions
from evaluation import eval_metrics
from model import _torch_model, _base_model, _tensorflow_model, _model_functions


class OptunaOptim:
    """
    Class that contains all info for the whole optimization using optuna for one model and dataset.

    ## Attributes ##
        task: str : ML task (regression or classification) depending on target variable
        current_model_name: str : name of the current model according to naming of .py file in package model
        dataset: base_dataset.Dataset : dataset to use for optimization run
        datasplit_subpath: str : subpath with datasplit info relevant for saving / naming
        base_path : str : base_path for save_path
        save_path: str : path for model and results storing
        study : optuna.study.Study : optuna study for optimization run
        current_best_val_result: float : the best validation result so far
        early_stopping_point: int : point at which early stopping occured (relevant for some models)
        user_input_params: dict : all params handed over to the constructor that are needed in the whole class
    """

    def __init__(self, save_dir: str, genotype_matrix_name: str, phenotype_matrix_name: str, phenotype: str,
                 n_outerfolds: int, n_innerfolds: int, val_set_size_percentage: int, test_set_size_percentage: int,
                 maf_percentage: int, n_trials: int, save_final_model: bool, batch_size: int, n_epochs: int,
                 task: str, current_model_name: str, dataset: base_dataset.Dataset, start_time: str):
        """
        Constructor of OptunaOptim.
        :param save_dir: directory for saving the results.
        :param genotype_matrix_name: name of the genotype matrix including datatype ending
        :param phenotype_matrix_name: name of the phenotype matrix including datatype ending
        :param phenotype: name of the phenotype to predict
        :param n_outerfolds: number of outerfolds relevant for nested-cv
        :param n_innerfolds: number of folds relevant for nested-cv and cv-test
        :param test_set_size_percentage: size of the test set relevant for cv-test and train-val-test
        :param val_set_size_percentage: size of the validation set relevant for train-val-test
        :param maf_percentage: threshold for MAF filter as percentage value
        :param n_trials: number of trials for optuna
        :param save_final_model: specify if the final model should be saved
        :param batch_size: batch size for neural network models
        :param n_epochs: number of epochs for neural network models
        :param task: ML task (regression or classification) depending on target variable
        :param current_model_name: name of the current model according to naming of .py file in package model
        :param dataset: dataset to use for optimization run
        :param start_time: starting time of the optimization run for saving purposes
        """
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
            phenotype_matrix_name.split('.')[0] + '/' + phenotype + '/' + self.dataset.datasplit + '/' + \
            self.datasplit_subpath + '/MAF' + str(maf_percentage) + '/' + start_time + '/' + current_model_name + '/'
        self.save_path = self.base_path
        self.study = None
        self.current_best_val_result = None
        self.early_stopping_point = None
        self.user_input_params = locals()  # distribute all handed over params in whole class

    def create_new_study(self) -> optuna.study.Study:
        """
        Create a new optuna study.
        :return: optuna study
        """
        outerfold_prefix = 'OUTER' + self.save_path[-2] + '-' if 'outerfold' in self.save_path else ''
        study_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + outerfold_prefix + \
                     self.user_input_params["genotype_matrix_name"].split('.')[0] + '-' + \
                     self.user_input_params["phenotype_matrix_name"].split('.')[0] + '-' + \
                     self.user_input_params["phenotype"] + '-MAF' + str(self.user_input_params["maf_percentage"]) + \
                     '-SPLIT' + self.dataset.datasplit + self.datasplit_subpath + \
                     '-MODEL' + self.current_model_name + '-TRIALS' + str(self.user_input_params["n_trials"])
        storage = optuna.storages.RDBStorage(
            "sqlite:////" + self.save_path + 'Optuna_DB-' + study_name + ".db", heartbeat_interval=60, grace_period=120,
            failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=3)
            )
        # TPE Sampler with seed for reproducibility
        # Percentile pruner if minimum 20 trials exist and intermediate result is worse than 80th percentile
        study = optuna.create_study(
            storage=storage, study_name=study_name,
            direction='minimize' if self.task == 'regression' else 'maximize', load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.PercentilePruner(percentile=80, n_min_trials=20)
        )
        return study

    def objective(self, trial: optuna.trial.Trial, train_val_indices: dict) -> float:
        """
        Objective function for optuna optimization that returns a score
        :param trial: trial of optuna for optimization
        :param train_val_indices: indices of train and validation sets
        :return: score of the current hyperparameter config
        """
        # Setup timers for runtime logging
        start_process_time = time.process_time()
        start_realclock_time = time.time()
        # Create model
        # in case a model has attributes not part of the base class hand them over in a dictionary to keep the same call
        # (name of the attribute and key in the dictionary have to match)
        additional_attributes_dict = {}
        if issubclass(utils.helper_functions.get_mapping_name_to_class()[self.current_model_name],
                      _torch_model.TorchModel) or \
                issubclass(utils.helper_functions.get_mapping_name_to_class()[self.current_model_name],
                           _tensorflow_model.TensorflowModel):
            # additional attributes for torch and tensorflow models
            additional_attributes_dict['n_features'] = self.dataset.X_full.shape[1]
            additional_attributes_dict['batch_size'] = self.user_input_params["batch_size"]
            additional_attributes_dict['n_epochs'] = self.user_input_params["n_epochs"]
            additional_attributes_dict['width_onehot'] = self.dataset.X_full.shape[-1]
            early_stopping_points = []  # log early stopping point at each fold for torch and tensorflow models
        try:
            model: _base_model.BaseModel = utils.helper_functions.get_mapping_name_to_class()[self.current_model_name](
                task=self.task, optuna_trial=trial,
                n_outputs=len(np.unique(self.dataset.y_full)) if self.task == 'classification' else 1,
                **additional_attributes_dict
            )
        except Exception as exc:
            print('Trial failed. Error in model creation.')
            print(exc)
            print(trial.params)
            self.clean_up_after_exception(trial_number=trial.number, trial_params=trial.params)
            raise optuna.exceptions.TrialPruned()

        # save the unfitted model
        os.makedirs(self.save_path + 'temp/', exist_ok=True)
        model.save_model(path=self.save_path + 'temp/',
                         filename='unfitted_model_trial' + str(trial.number))
        print("Params for Trial " + str(trial.number))
        print(trial.params)
        # Iterate over all innerfolds
        objective_values = []
        validation_results = pd.DataFrame(index=range(0, self.dataset.y_full.shape[0]))
        for innerfold_name, innerfold_info in train_val_indices.items():
            # skip test set (see structure described in base_dataset.Dataset)
            if innerfold_name == 'test':
                continue
            if self.dataset.datasplit != 'train-val-test':
                print('# Processing ' + innerfold_name + ' #')
            else:
                innerfold_name = 'train-val'
            # load the unfitted model to prevent information leak between folds
            model = _model_functions.load_model(path=self.save_path + 'temp/',
                                                filename='unfitted_model_trial' + str(trial.number))
            X_train, y_train, sample_ids_train, X_val, y_val, sample_ids_val = \
                self.dataset.X_full[innerfold_info['train']], \
                self.dataset.y_full[innerfold_info['train']], \
                self.dataset.sample_ids_full[innerfold_info['train']], \
                self.dataset.X_full[innerfold_info['val']], \
                self.dataset.y_full[innerfold_info['val']], \
                self.dataset.sample_ids_full[innerfold_info['val']]
            try:
                # run train and validation loop for this fold
                y_pred = model.train_val_loop(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
                if hasattr(model, 'early_stopping_point'):
                    early_stopping_points.append(
                        model.early_stopping_point if model.early_stopping_point is not None else model.n_epochs)
                if len(y_pred) == (len(y_val) - 1):
                    # might happen if batch size leads to a last batch with only one sample which will be dropped then
                    print('y_val has one element less than y_true (e.g. due to batch size config) -> drop last element')
                    y_val = y_val[:-1]
                objective_value = \
                    sklearn.metrics.accuracy_score(y_true=y_val, y_pred=y_pred) if self.task == 'classification' \
                    else sklearn.metrics.mean_squared_error(y_true=y_val, y_pred=y_pred)
                # report value for pruning
                trial.report(value=objective_value,
                             step=0 if self.dataset.datasplit == 'train-val-test' else int(innerfold_name[-1]))
                if trial.should_prune():
                    self.clean_up_after_exception(trial_number=trial.number, trial_params=trial.params)
                    raise optuna.exceptions.TrialPruned()
                # store results
                objective_values.append(objective_value)
                validation_results.at[0:len(sample_ids_train)-1, innerfold_name + '_train_sampleids'] = \
                    sample_ids_train.flatten()
                validation_results.at[0:len(y_train) - 1, innerfold_name + '_train_true'] = y_train.flatten()
                validation_results.at[0:len(y_train) - 1, innerfold_name + '_train_pred'] = \
                    model.predict(X_in=X_train).flatten()
                validation_results.at[0:len(sample_ids_val)-1, innerfold_name + '_val_sampleids'] = \
                    sample_ids_val.flatten()
                validation_results.at[0:len(y_val)-1, innerfold_name + '_val_true'] = y_val.flatten()
                validation_results.at[0:len(y_pred)-1, innerfold_name + '_val_pred'] = y_pred.flatten()
                for metric, value in eval_metrics.get_evaluation_report(y_pred=y_pred, y_true=y_val, task=self.task,
                                                                        prefix=innerfold_name + '_').items():
                    validation_results.at[0, metric] = value
            except (RuntimeError, TypeError, tf.errors.ResourceExhaustedError) as exc:
                print(exc)
                if 'out of memory' in str(exc) or isinstance(exc, tf.errors.ResourceExhaustedError):
                    # Recover from CUDA out of memory error
                    print('CUDA OOM at batch_size ' + str(model.batch_size))
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    print('Trial failed. Error in optim loop.')
                self.clean_up_after_exception(trial_number=trial.number, trial_params=trial.params)
                raise optuna.exceptions.TrialPruned()
        current_val_result = np.mean(objective_values)
        if self.current_best_val_result is None or \
                (self.task == 'classification' and current_val_result > self.current_best_val_result) or \
                (self.task == 'regression' and current_val_result < self.current_best_val_result):
            self.current_best_val_result = current_val_result
            if hasattr(model, 'early_stopping_point'):
                # take mean of early stopping points of all innerfolds for refitting of final model
                self.early_stopping_point = int(np.mean(early_stopping_points))
            # persist results
            validation_results.to_csv(self.save_path + 'temp/validation_results_trial' + str(trial.number) + '.csv',
                                      sep=',', decimal='.', float_format='%.10f', index=False)
            # delete previous results
            for file in os.listdir(self.save_path + 'temp/'):
                if 'trial' + str(trial.number) not in file:
                    os.remove(self.save_path + 'temp/' + file)
        else:
            # delete unfitted model
            os.remove(self.save_path + 'temp/' + 'unfitted_model_trial' + str(trial.number))
        # save runtime information of this trial
        self.write_runtime_csv(dict_runtime={'Trial': trial.number,
                                             'process_time_s': time.process_time() - start_process_time,
                                             'real_time_s': time.time() - start_realclock_time,
                                             'params': trial.params})
        return current_val_result

    def clean_up_after_exception(self, trial_number: int, trial_params: dict):
        """
        Clean up things after an exception: delete unfitted model if it exists and update runtime csv
        :param trial_number: number of the trial
        :param trial_params: parameters of the trial
        """
        if os.path.exists(self.save_path + 'temp/' + 'unfitted_model_trial' + str(trial_number)):
            os.remove(self.save_path + 'temp/' + 'unfitted_model_trial' + str(trial_number))
        self.write_runtime_csv(dict_runtime={'Trial': trial_number, 'process_time_s': np.nan, 'real_time_s': np.nan,
                                             'params': trial_params})

    def write_runtime_csv(self, dict_runtime: dict):
        """
        Write runtime info to runtime csv file
        :param dict_runtime: Dictionary with runtime information
        """
        with open(self.save_path + self.current_model_name + '_runtime_overview.csv', 'a') as runtime_file:
            headers = ['Trial', 'process_time_s', 'real_time_s', 'params']
            writer = csv.DictWriter(f=runtime_file, fieldnames=headers)
            if runtime_file.tell() == 0:
                writer.writeheader()
            writer.writerow(dict_runtime)

    def calc_runtime_stats(self) -> dict:
        """
        Calculate runtime stats for saved csv file.
        :return: dict with runtime info
        """
        csv_file = pd.read_csv(self.save_path + self.current_model_name + '_runtime_overview.csv')
        process_times = csv_file['process_time_s']
        real_times = csv_file['real_time_s']
        process_time_mean, process_time_std, process_time_max, process_time_min = \
            process_times.mean(), process_times.std(), process_times.max(), process_times.min()
        real_time_mean, real_time_std, real_time_max, real_time_min = \
            real_times.mean(), real_times.std(), real_times.max(), real_times.min()
        self.write_runtime_csv({'Trial': 'mean', 'process_time_s': process_time_mean, 'real_time_s': real_time_mean})
        self.write_runtime_csv({'Trial': 'std', 'process_time_s': process_time_std, 'real_time_s': real_time_std})
        self.write_runtime_csv({'Trial': 'max', 'process_time_s': process_time_max, 'real_time_s': real_time_max})
        self.write_runtime_csv({'Trial': 'min', 'process_time_s': process_time_min, 'real_time_s': real_time_min})
        return {'process_time_mean': process_time_mean, 'process_time_std': process_time_std,
                'process_time_max': process_time_max, 'process_time_min': process_time_min,
                'real_time_mean': real_time_mean, 'real_time_std': real_time_std,
                'real_time_max': real_time_max, 'real_time_min': real_time_min}

    def run_optuna_optimization(self) -> dict:
        """
        Function to run whole optuna optimization for one model, dataset and datasplit.
        """
        # Iterate over outerfolds
        # (according to structure described in base_dataset.Dataset, only for nested-cv multiple outerfolds exist)
        overall_results = {}
        for outerfold_name, outerfold_info in self.dataset.datasplit_indices.items():
            if self.dataset.datasplit == 'nested-cv':
                # Only print outerfold info for nested-cv as it does not apply for the other splits
                print("## Starting Optimization for " + outerfold_name + " ##")
                maf_ind = [m.end(0) for m in re.finditer(pattern='(MAF[0-9]+/)+([0-9]|-|_)+', string=self.base_path)][0]
                self.save_path = self.base_path[:maf_ind] + '/' + outerfold_name + self.base_path[maf_ind:]
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            # Create a new study for each outerfold
            self.study = self.create_new_study()
            self.current_best_val_result = None
            # Start optimization run
            self.study.optimize(
                lambda trial: self.objective(trial=trial, train_val_indices=outerfold_info),
                n_trials=self.user_input_params["n_trials"]
            )
            # Calculate runtime metrics after finishing optimization
            runtime_metrics = self.calc_runtime_stats()
            # Print statistics after run
            print("## Optuna Study finished ##")
            print("Study statistics: ")
            print("  Finished trials: ", len(self.study.trials))
            print("  Pruned trials: ", len(self.study.get_trials(states=(optuna.trial.TrialState.PRUNED,))))
            print("  Completed trials: ", len(self.study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))))
            print("  Best Trial: ", self.study.best_trial.number)
            print("  Value: ", self.study.best_trial.value)
            print("  Params: ")
            for key, value in self.study.best_trial.params.items():
                print("    {}: {}".format(key, value))

            # Move validation results and models of best trial
            files_to_keep = glob.glob(self.save_path + 'temp/' + '*trial' + str(self.study.best_trial.number) + '*')
            for file in files_to_keep:
                shutil.copyfile(file, self.save_path + file.split('/')[-1])
            shutil.rmtree(self.save_path + 'temp/')

            # Retrain on full train + val data with best hyperparams and apply on test
            print("## Retrain best model and test ##")
            X_test, y_test, sample_ids_test = \
                self.dataset.X_full[outerfold_info['test']], self.dataset.y_full[outerfold_info['test']], \
                self.dataset.sample_ids_full[outerfold_info['test']]
            X_retrain, y_retrain, sample_ids_retrain = \
                self.dataset.X_full[~np.isin(np.arange(len(self.dataset.X_full)), outerfold_info['test'])], \
                self.dataset.y_full[~np.isin(np.arange(len(self.dataset.y_full)), outerfold_info['test'])], \
                self.dataset.sample_ids_full[~np.isin(np.arange(len(self.dataset.sample_ids_full)),
                                                      outerfold_info['test'])],
            start_process_time = time.process_time()
            start_realclock_time = time.time()
            final_model = _model_functions.load_retrain_model(
                path=self.save_path, filename='unfitted_model_trial' + str(self.study.best_trial.number),
                X_retrain=X_retrain, y_retrain=y_retrain, early_stopping_point=self.early_stopping_point)
            y_pred_retrain = final_model.predict(X_in=X_retrain)
            self.write_runtime_csv(dict_runtime={'Trial': 'retraining',
                                                 'process_time_s': time.process_time() - start_process_time,
                                                 'real_time_s': time.time() - start_realclock_time})
            y_pred_test = final_model.predict(X_in=X_test)

            # Evaluate and save results
            eval_scores = \
                eval_metrics.get_evaluation_report(y_pred=y_pred_test, y_true=y_test, task=self.task, prefix='test_')
            key = outerfold_name if self.dataset.datasplit == 'nested-cv' else 'Test'
            overall_results[key] = {'best_params': self.study.best_trial.params, 'eval_metrics': eval_scores,
                                    'runtime_metrics': runtime_metrics}
            print('## Results on test set ##')
            print(eval_scores)
            final_results = pd.DataFrame(index=range(0, self.dataset.y_full.shape[0]))
            final_results.at[0:len(sample_ids_retrain)-1, 'sample_ids_retrain'] = sample_ids_retrain.flatten()
            final_results.at[0:len(y_pred_retrain)-1, 'y_pred_retrain'] = y_pred_retrain.flatten()
            final_results.at[0:len(y_retrain)-1, 'y_true_retrain'] = y_retrain.flatten()
            final_results.at[0:len(sample_ids_test)-1, 'sample_ids_test'] = sample_ids_test.flatten()
            final_results.at[0:len(y_pred_test)-1, 'y_pred_test'] = y_pred_test.flatten()
            final_results.at[0:len(y_test)-1, 'y_true_test'] = y_test.flatten()
            for metric, value in eval_scores.items():
                final_results.at[0, metric] = value
            final_results.to_csv(self.save_path + 'final_model_test_results.csv',
                                 sep=',', decimal='.', float_format='%.10f', index=False)
            if self.user_input_params["save_final_model"]:
                final_model.save_model(path=self.save_path,
                                       filename='final_retrained_model')
        return overall_results
