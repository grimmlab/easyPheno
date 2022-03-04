import argparse
import datetime
import optuna
import pandas as pd
import sklearn
import numpy as np
import os
import glob
import shutil

import utils
from preprocess import base_dataset
from utils import helper_functions
from evaluation import eval_metrics
from model import torch_model, base_model


class OptunaOptim:
    """
    Class that contains all info for the whole optimization using optuna for one model and dataset

    ## Attributes ##
        # Instance attributes #
        arguments: argparse.Namespace : all arguments provided by the user
        task: str : ML task (regression or classification) depending on target variable
        current_model_name: str : name of the current model according to naming of .py file in package model
        dataset: base_dataset.Dataset : dataset to use for optimization run
        base_path : str : base_path for save_path
        save_path: str : path for model and results storing
        study : optuna.study.Study : optuna study for optimization run
    """

    def __init__(self, arguments: argparse.Namespace, task: str, current_model_name: str,
                 dataset: base_dataset.Dataset):
        """
        Constructor of OptunaOptim
        :param arguments: all arguments provided by the user
        :param task: ML task (regression or classification) depending on target variable
        :param current_model_name: name of the current model according to naming of .py file in package model
        :param dataset: dataset to use for optimization run
        """
        self.current_model_name = current_model_name
        self.task = task
        self.arguments = arguments
        self.dataset = dataset
        self.base_path = arguments.save_dir + \
            '/results/' + arguments.genotype_matrix.split('.')[0] + \
            '/' + arguments.phenotype_matrix.split('.')[0] + '/' + arguments.phenotype + \
            '/' + current_model_name + '/' + arguments.datasplit + '/' + \
            helper_functions.get_subpath_for_datasplit(arguments=arguments, datasplit=arguments.datasplit) + '/' + \
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '/'
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        self.save_path = self.base_path
        self.study = None
        self.current_best_val_result = None

    def create_new_study(self) -> optuna.study.Study:
        """
        Method to create a new optuna study
        :return: optuna study
        """
        outerfold_prefix = 'OUTER' + self.save_path[-2] + '-' if 'outerfold' in self.save_path else ''
        study_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + outerfold_prefix + \
                     self.arguments.genotype_matrix.split('.')[0] + '-' + \
                     self.arguments.phenotype_matrix.split('.')[0] + '-' + self.arguments.phenotype + '-' + \
                     'MAF' + str(self.arguments.maf_percentage) + \
                     '-SPLIT' + self.arguments.datasplit + \
                     helper_functions.get_subpath_for_datasplit(arguments=self.arguments,
                                                                datasplit=self.arguments.datasplit) + \
                     '-MODEL' + self.arguments.model + '-TRIALS' + str(self.arguments.n_trials)
        storage = optuna.storages.RDBStorage(
            "sqlite:////" + self.save_path + 'Optuna_DB-' + study_name + ".db", heartbeat_interval=60, grace_period=120,
            failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=3)
            )
        study = optuna.create_study(
            storage=storage, study_name=study_name,
            direction='minimize' if self.task == 'regression' else 'maximize', load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.PercentilePruner(percentile=80, n_min_trials=20)
        )

        return study

    def objective(self, trial: optuna.trial.Trial, train_val_indices: dict):
        """
        Objective function for optuna optimization that returns a score
        :param trial: trial of optuna for optimization
        :param train_val_indices: indices of train and validation sets
        :return: score of the current hyperparameter config
        """
        # Create model
        # in case a model has attributes not part of the base class hand them over in a dictionary to keep the same call
        # (name of the attribute and key in the dictionary have to match)
        additional_attributes_dict = {}
        if issubclass(utils.helper_functions.get_mapping_name_to_class()[self.current_model_name],
                      torch_model.TorchModel):
            # all torch models have the number of input features as attribute
            additional_attributes_dict['n_features'] = self.dataset.X_full.shape[1]
            additional_attributes_dict['batch_size'] = self.arguments.batch_size
            additional_attributes_dict['n_epochs'] = self.arguments.n_epochs
        model: base_model.BaseModel = utils.helper_functions.get_mapping_name_to_class()[self.current_model_name](
            task=self.task, optuna_trial=trial,
            n_outputs=len(np.unique(self.dataset.y_full)) if self.task == 'classification' else 1,
            **additional_attributes_dict
        )
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
            model = base_model.load_model(path=self.save_path + 'temp/',
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
                objective_value = \
                    sklearn.metrics.accuracy_score(y_true=y_val, y_pred=y_pred) if self.task == 'classification' \
                    else sklearn.metrics.mean_squared_error(y_true=y_val, y_pred=y_pred)
                # report value for pruning
                # step has an offset based on outerfold_number as same study is used for all outerfolds
                trial.report(value=objective_value,
                             step=0 if self.dataset.datasplit == 'train-val-test' else int(innerfold_name[-1]))
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                # store results
                objective_values.append(objective_value)
                validation_results.at[0:len(sample_ids_train)-1, innerfold_name + '_train_sampleids'] = \
                    sample_ids_train.flatten()
                validation_results.at[0:len(y_train) - 1, innerfold_name + '_train_true'] = y_train.flatten()
                validation_results.at[0:len(y_train) - 1, innerfold_name + '_train_pred'] = \
                    model.predict(X_in=X_train).flatten()
                validation_results.at[0:len(sample_ids_val)-1, innerfold_name + '_val_sampleids'] = sample_ids_val.flatten()
                validation_results.at[0:len(y_val)-1, innerfold_name + '_val_true'] = y_val.flatten()
                validation_results.at[0:len(y_pred)-1, innerfold_name + '_val_pred'] = y_pred.flatten()
                for metric, value in eval_metrics.get_evaluation_report(y_pred=y_pred, y_true=y_val, task=self.task,
                                                                        prefix=innerfold_name + '_').items():
                    validation_results.at[0, metric] = value
                # model.save_model(path=self.save_path + 'temp/',
                #                 filename=innerfold_name + '-validation_model_trial' + str(trial.number))
            except Exception as exc:
                print('Trial failed')
                print(exc)
                break
        current_val_result = np.mean(objective_values)
        if self.current_best_val_result is None or \
                (self.task == 'classification' and current_val_result > self.current_best_val_result) or \
                (self.task == 'regression' and current_val_result < self.current_best_val_result):
            self.current_best_val_result = current_val_result
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

        return current_val_result

    def run_optuna_optimization(self):
        """
        Function to run whole optuna optimization for one model, dataset and datasplit
        """

        # Iterate over outerfolds
        # (according to structure described in base_dataset.Dataset, only for nested-cv multiple outerfolds exist)
        for outerfold_name, outerfold_info in self.dataset.datasplit_indices.items():
            if self.dataset.datasplit == 'nested-cv':
                # Only print outerfold info for nested-cv as it does not apply for the other splits
                print("## Starting Optimization for " + outerfold_name + " ##")
                self.save_path = self.base_path + outerfold_name + '/'
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
            # Create a new study for each outerfold
            self.study = self.create_new_study()
            self.current_best_val_result = None
            # Start optimization run
            self.study.optimize(
                lambda trial: self.objective(trial=trial, train_val_indices=outerfold_info),
                n_trials=self.arguments.n_trials
            )
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

            print("## Retrain best model and test ##")
            # Retrain on full train + val data with best hyperparams and apply on test
            X_test, y_test, sample_ids_test = \
                self.dataset.X_full[outerfold_info['test']], self.dataset.y_full[outerfold_info['test']], \
                self.dataset.sample_ids_full[outerfold_info['test']]
            X_retrain, y_retrain, sample_ids_retrain = \
                self.dataset.X_full[~np.isin(np.arange(len(self.dataset.X_full)), outerfold_info['test'])], \
                self.dataset.y_full[~np.isin(np.arange(len(self.dataset.y_full)), outerfold_info['test'])], \
                self.dataset.sample_ids_full[~np.isin(np.arange(len(self.dataset.sample_ids_full)),
                                                      outerfold_info['test'])],
            final_model = base_model.load_retrain_model(
                path=self.save_path, filename='unfitted_model_trial' + str(self.study.best_trial.number),
                X_retrain=X_retrain, y_retrain=y_retrain)
            y_pred_retrain = final_model.predict(X_in=X_retrain)
            y_pred_test = final_model.predict(X_in=X_test)

            # Evaluate and save results
            eval_scores = \
                eval_metrics.get_evaluation_report(y_pred=y_pred_test, y_true=y_test, task=self.task, prefix='test_')
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
            if self.arguments.save_final_model:
                final_model.save_model(path=self.save_path,
                                       filename='final_retrained_model')