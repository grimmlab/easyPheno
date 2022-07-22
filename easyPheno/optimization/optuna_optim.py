import datetime
import optuna
import pandas as pd
import sklearn
import sklearn.inspection
import numpy as np
import shutil
import time
import csv
import gc
import pathlib

import torch.cuda
import tensorflow as tf

from ..preprocess import base_dataset
from ..utils import helper_functions
from ..evaluation import eval_metrics
from ..model import _base_model, _model_functions, _tensorflow_model, _torch_model


class OptunaOptim:
    """
    Class that contains all info for the whole optimization using optuna for one model and dataset.

    **Attributes**

        - task (*str*): ML task (regression or classification) depending on target variable
        - current_model_name (*str*): name of the current model according to naming of .py file in package model
        - dataset (:obj:`~easyPheno.preprocess.base_dataset.Dataset`): dataset to use for optimization run
        - datasplit_subpath (*str*): subpath with datasplit info relevant for saving / naming
        - base_path (*str*): base_path for save_path
        - save_path (*str*): path for model and results storing
        - study (*optuna.study.Study*): optuna study for optimization run
        - current_best_val_result (*float*): the best validation result so far
        - early_stopping_point (*int*): point at which early stopping occured (relevant for some models)
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
    :param n_trials: number of trials for optuna
    :param save_final_model: specify if the final model should be saved
    :param batch_size: batch size for neural network models
    :param n_epochs: number of epochs for neural network models
    :param task: ML task (regression or classification) depending on target variable
    :param current_model_name: name of the current model according to naming of .py file in package model
    :param dataset: dataset to use for optimization run
    :param models_start_time: optimized models and starting time of the optimization run for saving purposes
    :param intermediate_results_interval: number of trials after which intermediate results will be saved
    """

    def __init__(self, save_dir: pathlib.Path, genotype_matrix_name: str, phenotype_matrix_name: str, phenotype: str,
                 n_outerfolds: int, n_innerfolds: int, val_set_size_percentage: int, test_set_size_percentage: int,
                 maf_percentage: int, n_trials: int, save_final_model: bool, batch_size: int, n_epochs: int,
                 task: str, current_model_name: str, dataset: base_dataset.Dataset, models_start_time: str,
                 intermediate_results_interval: int = 50):
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
        self.base_path = save_dir.joinpath('results', genotype_matrix_name.split('.')[0], \
                         phenotype_matrix_name.split('.')[0], phenotype, self.dataset.datasplit + '_' + \
                         self.datasplit_subpath + '_MAF' + str(maf_percentage) + '_' + models_start_time, \
                         current_model_name)
        self.save_path = self.base_path
        self.study = None
        self.current_best_val_result = None
        self.early_stopping_point = None
        self.intermediate_results_interval = intermediate_results_interval
        self.user_input_params = locals()  # distribute all handed over params in whole class

    def create_new_study(self) -> optuna.study.Study:
        """
        Create a new optuna study.

        :return: a new optuna study instance
        """
        # outerfold_prefix = \
        #    'OUTER' + self.save_path[[m.end(0) for m in re.finditer(pattern='outerfold_', string=self.save_path)][0]] \
        #    + '-' if 'outerfold' in self.save_path else ''
        outerfold_prefix = \
            'OUTER' + self.save_path.parts[-2].split('_')[1] + '-' if 'outerfold' in self.save_path.parts[-2] else ''
        study_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + outerfold_prefix + \
                     self.user_input_params["genotype_matrix_name"].split('.')[0] + '-' + \
                     self.user_input_params["phenotype_matrix_name"].split('.')[0] + '-' + \
                     self.user_input_params["phenotype"] + '-MAF' + str(self.user_input_params["maf_percentage"]) + \
                     '-SPLIT' + self.dataset.datasplit + self.datasplit_subpath + \
                     '-MODEL' + self.current_model_name + '-TRIALS' + str(self.user_input_params["n_trials"])
        storage = optuna.storages.RDBStorage(
            "sqlite:////" + str(self.save_path.joinpath('Optuna_DB.db')), heartbeat_interval=60, grace_period=120,
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
        if (trial.number != 0) and (trial.number % self.intermediate_results_interval == 0):
            print('Generate intermediate test results at trial ' + str(trial.number))
            _ = self.generate_results_on_test(outerfold_info=train_val_indices)

        # Setup timers for runtime logging
        start_process_time = time.process_time()
        start_realclock_time = time.time()
        # Create model
        # in case a model has attributes not part of the base class hand them over in a dictionary to keep the same call
        # (name of the attribute and key in the dictionary have to match)
        additional_attributes_dict = {}
        if issubclass(helper_functions.get_mapping_name_to_class()[self.current_model_name],
                      _torch_model.TorchModel) or \
                issubclass(helper_functions.get_mapping_name_to_class()[self.current_model_name],
                           _tensorflow_model.TensorflowModel):
            # additional attributes for torch and tensorflow models
            additional_attributes_dict['n_features'] = self.dataset.X_full.shape[1]
            additional_attributes_dict['batch_size'] = self.user_input_params["batch_size"]
            additional_attributes_dict['n_epochs'] = self.user_input_params["n_epochs"]
            additional_attributes_dict['width_onehot'] = self.dataset.X_full.shape[-1]
            early_stopping_points = []  # log early stopping point at each fold for torch and tensorflow models
        try:
            model: _base_model.BaseModel = helper_functions.get_mapping_name_to_class()[self.current_model_name](
                task=self.task, optuna_trial=trial,
                n_outputs=len(np.unique(self.dataset.y_full)) if self.task == 'classification' else 1,
                **additional_attributes_dict
            )
        except Exception as exc:
            print('Trial failed. Error in model creation.')
            print(exc)
            print(trial.params)
            self.clean_up_after_exception(trial_number=trial.number, trial_params=trial.params,
                                          reason='model creation: ' + str(exc))
            raise optuna.exceptions.TrialPruned()

        # save the unfitted model
        self.save_path.joinpath('temp').mkdir(parents=True, exist_ok=True)
        model.save_model(path=self.save_path.joinpath('temp'),
                         filename='unfitted_model_trial' + str(trial.number))
        print("Params for Trial " + str(trial.number))
        print(trial.params)
        if self.check_params_for_duplicate(current_params=trial.params):
            print('Trial params are a duplicate.')
            self.clean_up_after_exception(trial_number=trial.number, trial_params=trial.params,
                                          reason='pruned: duplicate')
            raise optuna.exceptions.TrialPruned()
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
            model = _model_functions.load_model(path=self.save_path.joinpath('temp'),
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
                    self.clean_up_after_exception(trial_number=trial.number, trial_params=trial.params, reason='pruned')
                    raise optuna.exceptions.TrialPruned()
                # store results
                objective_values.append(objective_value)
                validation_results.at[0:len(sample_ids_train) - 1, innerfold_name + '_train_sampleids'] = \
                    sample_ids_train.flatten()
                validation_results.at[0:len(y_train) - 1, innerfold_name + '_train_true'] = y_train.flatten()
                y_train_pred = model.predict(X_in=X_train)
                validation_results.at[0:len(y_train_pred) - 1, innerfold_name + '_train_pred'] = \
                    y_train_pred.flatten()
                validation_results.at[0:len(sample_ids_val) - 1, innerfold_name + '_val_sampleids'] = \
                    sample_ids_val.flatten()
                validation_results.at[0:len(y_val) - 1, innerfold_name + '_val_true'] = y_val.flatten()
                validation_results.at[0:len(y_pred) - 1, innerfold_name + '_val_pred'] = y_pred.flatten()
                for metric, value in eval_metrics.get_evaluation_report(y_pred=y_pred, y_true=y_val, task=self.task,
                                                                        prefix=innerfold_name + '_').items():
                    validation_results.at[0, metric] = value
            except (RuntimeError, TypeError, tf.errors.ResourceExhaustedError, ValueError) as exc:
                print(exc)
                if 'out of memory' in str(exc) or isinstance(exc, tf.errors.ResourceExhaustedError):
                    # Recover from CUDA out of memory error
                    print('CUDA OOM at batch_size ' + str(model.batch_size))
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    print('Trial failed. Error in optim loop.')
                self.clean_up_after_exception(trial_number=trial.number, trial_params=trial.params,
                                              reason='model optimization: ' + str(exc))
                raise optuna.exceptions.TrialPruned()
        current_val_result = float(np.mean(objective_values))
        if self.current_best_val_result is None or \
                (self.task == 'classification' and current_val_result > self.current_best_val_result) or \
                (self.task == 'regression' and current_val_result < self.current_best_val_result):
            self.current_best_val_result = current_val_result
            if hasattr(model, 'early_stopping_point'):
                # take mean of early stopping points of all innerfolds for refitting of final model
                self.early_stopping_point = int(np.mean(early_stopping_points))
            # persist results
            validation_results.to_csv(self.save_path.joinpath('temp/validation_results_trial' + str(trial.number) +
                                                    '.csv'), sep=',', decimal='.', float_format='%.10f', index=False)
            # delete previous results
            for file in self.save_path.joinpath('temp').iterdir():
                if 'trial' + str(trial.number) not in str(file):
                    self.save_path.joinpath('temp', file).unlink()
        else:
            # delete unfitted model
            self.save_path.joinpath('temp', 'unfitted_model_trial' + str(trial.number)).unlink()
        # save runtime information of this trial
        self.write_runtime_csv(dict_runtime={'Trial': trial.number,
                                             'process_time_s': time.process_time() - start_process_time,
                                             'real_time_s': time.time() - start_realclock_time,
                                             'params': trial.params, 'note': 'successful'})
        return current_val_result

    def clean_up_after_exception(self, trial_number: int, trial_params: dict, reason: str):
        """
        Clean up things after an exception: delete unfitted model if it exists and update runtime csv

        :param trial_number: number of the trial
        :param trial_params: parameters of the trial
        :param reason: hint for the reason of the Exception
        """
        if self.save_path.joinpath('temp', 'unfitted_model_trial' + str(trial_number)).exists():
            self.save_path.joinpath('temp', 'unfitted_model_trial' + str(trial_number)).unlink()
        self.write_runtime_csv(dict_runtime={'Trial': trial_number, 'process_time_s': np.nan, 'real_time_s': np.nan,
                                             'params': trial_params, 'note': reason})

    def write_runtime_csv(self, dict_runtime: dict):
        """
        Write runtime info to runtime csv file

        :param dict_runtime: dictionary with runtime information
        """
        with open(self.save_path.joinpath(self.current_model_name + '_runtime_overview.csv'), 'a') as runtime_file:
            headers = ['Trial', 'process_time_s', 'real_time_s', 'params', 'note']
            writer = csv.DictWriter(f=runtime_file, fieldnames=headers)
            if runtime_file.tell() == 0:
                writer.writeheader()
            writer.writerow(dict_runtime)

    def calc_runtime_stats(self) -> dict:
        """
        Calculate runtime stats for saved csv file.

        :return: dict with runtime info enhanced with runtime stats
        """
        csv_file = pd.read_csv(self.save_path.joinpath(self.current_model_name + '_runtime_overview.csv'))
        if csv_file['Trial'].dtype is object and any(["retrain" in elem for elem in csv_file["Trial"]]):
            csv_file = csv_file[csv_file["Trial"].str.contains("retrain") == False]
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

    def check_params_for_duplicate(self, current_params: dict) -> bool:
        """
        Check if params were already suggested which might happen by design of TPE sampler.

        :param current_params: dictionar with current parameters

        :return: bool reflecting if current params were already used in the same study
        """
        past_params = [trial.params for trial in self.study.trials[:-1]]
        return current_params in past_params

    def generate_results_on_test(self, outerfold_info: dict) -> dict:
        """
        Generate the results on the testing data

        :param outerfold_info: dictionary with outerfold datasplit indices

        :return: evaluation metrics dictionary
        """

        helper_functions.set_all_seeds()
        # Retrain on full train + val data with best hyperparams and apply on test
        print("## Retrain best model and test ##")
        X_test, y_test, sample_ids_test = \
            self.dataset.X_full[outerfold_info['test']], self.dataset.y_full[outerfold_info['test']], \
            self.dataset.sample_ids_full[outerfold_info['test']]
        X_retrain, y_retrain, sample_ids_retrain = \
            self.dataset.X_full[~np.isin(np.arange(len(self.dataset.X_full)), outerfold_info['test'])], \
            self.dataset.y_full[~np.isin(np.arange(len(self.dataset.y_full)), outerfold_info['test'])], \
            self.dataset.sample_ids_full[~np.isin(np.arange(len(self.dataset.sample_ids_full)),
                                                  outerfold_info['test'])]
        start_process_time = time.process_time()
        start_realclock_time = time.time()
        prefix = '' if len(self.study.trials) == self.user_input_params["n_trials"] else '/temp/'
        final_model = _model_functions.load_retrain_model(
            path=self.save_path, filename=prefix + 'unfitted_model_trial' + str(self.study.best_trial.number),
            X_retrain=X_retrain, y_retrain=y_retrain, early_stopping_point=self.early_stopping_point)
        y_pred_retrain = final_model.predict(X_in=X_retrain)
        no_trials = len(self.study.trials) - 1 if len(self.study.trials) % self.intermediate_results_interval != 0 \
            else len(self.study.trials)
        self.write_runtime_csv(dict_runtime={'Trial': 'retraining_after_' + str(no_trials),
                                             'process_time_s': time.process_time() - start_process_time,
                                             'real_time_s': time.time() - start_realclock_time,
                                             'params': self.study.best_trial.params, 'note': 'successful'})
        y_pred_test = final_model.predict(X_in=X_test)

        # Evaluate and save results
        eval_scores = \
            eval_metrics.get_evaluation_report(y_pred=y_pred_test, y_true=y_test, task=self.task, prefix='test_')

        feat_import_df = None
        if self.current_model_name in ['randomforest', 'xgboost', 'linearregression', 'elasticnet']:
            feat_import_df = self.get_feature_importance(model=final_model, X=X_test, y=y_test)

        print('## Results on test set ##')
        print(eval_scores)
        final_results = pd.DataFrame(index=range(0, self.dataset.y_full.shape[0]))
        final_results.at[0:len(sample_ids_retrain) - 1, 'sample_ids_retrain'] = sample_ids_retrain.flatten()
        final_results.at[0:len(y_pred_retrain) - 1, 'y_pred_retrain'] = y_pred_retrain.flatten()
        final_results.at[0:len(y_retrain) - 1, 'y_true_retrain'] = y_retrain.flatten()
        final_results.at[0:len(sample_ids_test) - 1, 'sample_ids_test'] = sample_ids_test.flatten()
        final_results.at[0:len(y_pred_test) - 1, 'y_pred_test'] = y_pred_test.flatten()
        final_results.at[0:len(y_test) - 1, 'y_true_test'] = y_test.flatten()
        for metric, value in eval_scores.items():
            final_results.at[0, metric] = value
        if len(self.study.trials) == self.user_input_params["n_trials"]:
            results_filename = 'final_model_test_results.csv'
            feat_import_filename = 'final_model_feature_importances.csv'
            if self.user_input_params["save_final_model"]:
                final_model.save_model(path=self.save_path, filename='final_retrained_model')
        else:
            results_filename = '/temp/intermediate_after_' + str(len(self.study.trials) - 1) + '_test_results.csv'
            feat_import_filename = \
                '/temp/intermediate_after_' + str(len(self.study.trials) - 1) + '_feat_importances.csv'
            shutil.copyfile(self.save_path.joinpath(self.current_model_name + '_runtime_overview.csv'),
                            self.save_path.joinpath('/temp/intermediate_after_' + str(len(self.study.trials) - 1) + '_' +
                            self.current_model_name + '_runtime_overview.csv'), )
        final_results.to_csv(
            self.save_path.joinpath(results_filename), sep=',', decimal='.', float_format='%.10f', index=False
        )
        if feat_import_df is not None:
            feat_import_df.to_csv(
                self.save_path.joinpath(feat_import_filename), sep=',', decimal='.', float_format='%.10f', index=False
            )
        return eval_scores

    def get_feature_importance(self, model: _base_model.BaseModel, X: np.array, y: np.array,
                               top_n: int = 1000, include_perm_importance: bool = False) -> pd.DataFrame:
        """
        Get feature importances for models that possess such a feature, e.g. XGBoost

        :param model: model to analyze
        :param X: feature matrix for permutation
        :param y: target vector for permutation
        :param top_n: top n features to select
        :param include_perm_importance: include permutation based feature importance or not

        :return: DataFrame with feature importance information
        """

        top_n = min(len(self.dataset.snp_ids), top_n)
        feat_import_df = pd.DataFrame()
        if self.current_model_name in ['randomforest', 'xgboost']:
            feature_importances = model.model.feature_importances_
            sorted_idx = feature_importances.argsort()[::-1][:top_n]
            feat_import_df['snp_ids_standard'] = self.dataset.snp_ids[sorted_idx]
            feat_import_df['feat_importance_standard'] = feature_importances[sorted_idx]
        else:
            coefs = model.model.coef_
            dims = coefs.shape[0] if len(coefs.shape) > 1 else 1
            for dim in range(dims):
                coef = coefs[dim] if len(coefs.shape) > 1 else coefs
                sorted_idx = coef.argsort()[::-1][:top_n]
                feat_import_df['snp_ids_' + str(dim)] = self.dataset.snp_ids[sorted_idx]
                feat_import_df['coefficients_' + str(dim)] = coef[sorted_idx]
        if include_perm_importance:
            perm_importance = sklearn.inspection.permutation_importance(
                estimator=model.model, X=X, y=y
            )
            sorted_idx = perm_importance.importances_mean.argsort()[::-1][:top_n]
            feat_import_df['snp_ids_perm'] = self.dataset.snp_ids[sorted_idx]
            feat_import_df['feat_importance_perm_mean'] = perm_importance.importances_mean[sorted_idx]
            feat_import_df['feat_importance_perm_std'] = perm_importance.importances_std[sorted_idx]

        return feat_import_df

    def run_optuna_optimization(self) -> dict:
        """
        Run whole optuna optimization for one model, dataset and datasplit.

        :return: dictionary with results overview
        """
        # Iterate over outerfolds
        # (according to structure described in base_dataset.Dataset, only for nested-cv multiple outerfolds exist)
        helper_functions.set_all_seeds()
        overall_results = {}
        for outerfold_name, outerfold_info in self.dataset.datasplit_indices.items():
            if self.dataset.datasplit == 'nested-cv':
                # Only print outerfold info for nested-cv as it does not apply for the other splits
                print("## Starting Optimization for " + outerfold_name + " ##")
                #end_ind = [m.end(0) for m in re.finditer(pattern='/', string=self.base_path)][-2]
                #self.save_path = self.base_path[:end_ind] + outerfold_name + '/' + self.base_path[end_ind:]
                self.save_path = self.base_path.parent.joinpath(outerfold_name, self.base_path.name)
            if not self.save_path.exists():
                self.save_path.mkdir(parents=True, exist_ok=True)
            # Create a new study for each outerfold
            self.study = self.create_new_study()
            self.current_best_val_result = None
            # Start optimization run
            self.study.optimize(
                lambda trial: self.objective(trial=trial, train_val_indices=outerfold_info),
                n_trials=self.user_input_params["n_trials"]
            )
            helper_functions.set_all_seeds()
            # set seeds again after optuna runs are finished as number of trials might lead to different states of
            # random number generators
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
            # files_to_keep = glob.glob(self.save_path + 'temp/' + '*trial' + str(self.study.best_trial.number) + '*')
            files_to_keep_path = self.save_path.joinpath('temp', '*trial' + str(self.study.best_trial.number) + '*')
            files_to_keep = pathlib.Path(files_to_keep_path.parent).expanduser().glob(files_to_keep_path.name)
            for file in files_to_keep:
                shutil.copyfile(file, self.save_path.joinpath(file.name))
            shutil.rmtree(self.save_path.joinpath('temp'))

            # Retrain on full train + val data with best hyperparams and apply on test
            eval_scores = self.generate_results_on_test(outerfold_info=outerfold_info)
            key = outerfold_name if self.dataset.datasplit == 'nested-cv' else 'Test'
            best_params = self.study.best_trial.params
            if issubclass(helper_functions.get_mapping_name_to_class()[self.current_model_name],
                          _torch_model.TorchModel) or \
                    issubclass(helper_functions.get_mapping_name_to_class()[self.current_model_name],
                               _tensorflow_model.TensorflowModel):
                # additional attributes for torch and tensorflow models
                if 'n_epochs' not in best_params.keys():
                    best_params['n_epochs'] = self.user_input_params["n_epochs"]
                if 'batch_size' not in best_params.keys():
                    best_params['batch_size'] = self.user_input_params["batch_size"]
                best_params['early_stopping_point'] = self.early_stopping_point

            overall_results[key] = {'best_params': best_params, 'eval_metrics': eval_scores,
                                    'runtime_metrics': runtime_metrics}
        return overall_results
