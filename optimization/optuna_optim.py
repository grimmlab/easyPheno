import argparse
import datetime
import optuna
import pandas as pd
import sklearn
import numpy as np
import os
import glob
import shutil
import joblib

import utils
from preprocess import base_dataset
from utils import helper_functions
from evaluation import eval_metrics


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
        self.base_path = arguments.base_dir + 'results/' + \
            arguments.genotype_matrix + '/' + arguments.phenotype_matrix + '/' + arguments.phenotype + \
            '/' + current_model_name + '/' + helper_functions.get_subpath_for_datasplit(arguments=arguments) + '/' + \
            str(datetime.datetime.now()) + '/'
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        self.save_path = self.base_path
        self.study = None

    def create_new_study(self) -> optuna.study.Study:
        """
        Method to create a new optuna study
        :return: optuna study
        """
        outerfold_suffix = '-OUTER' + self.save_path[-2] if 'outerfold' in self.save_path else ''
        study_name = str(datetime.datetime.now()) + '_' + self.arguments.genotype_matrix + '-' + \
                     self.arguments.phenotype_matrix + '-' + self.arguments.phenotype + '-' + \
                     'MAF' + str(self.arguments.maf_percentage) + \
                     '-SPLIT' + self.arguments.datasplit + \
                     helper_functions.get_subpath_for_datasplit(arguments=self.arguments) + \
                     '-MODEL' + self.arguments.model + '-TRIALS' + str(self.arguments.n_trials) + outerfold_suffix
        storage = optuna.storages.RDBStorage(
            "sqlite:////" + self.save_path + study_name + ".db", heartbeat_interval=10
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
        model = utils.helper_functions.get_mapping_name_to_class()[self.current_model](
            task=self.task, optuna_trial=trial, **additional_attributes_dict
        )
        # save the unfitted model
        os.makedirs(self.save_path + 'temp/')
        model.save_model(path=self.save_path + 'temp/',
                         filename='unfitted_model_trial' + str(trial.number))

        # Iterate over all innerfolds
        objective_values = []
        validation_results = {}
        for innerfold_name, innerfold_info in train_val_indices.items():
            # load the unfitted model to prevent information leak between folds
            model = joblib.load(self.save_path + 'temp/' + 'unfitted_model_trial' + str(trial.number))
            # skip test set (see structure described in base_dataset.Dataset)
            if innerfold_name == 'test':
                continue
            if self.dataset.datasplit != 'train-val-test':
                print('# Processing ' + innerfold_name + ' #')
            else:
                innerfold_name = 'train-val'
            X_train, y_train, X_val, y_val = \
                self.dataset.X_full[innerfold_info['train']], self.dataset.y_full[innerfold_info['train']], \
                self.dataset.X_full[innerfold_info['val']], self.dataset.y_full[innerfold_info['val']]
            # train model
            model.train(X_train=X_train, y_train=y_train)
            # validate model
            y_pred = model.predict(X_in=X_val)
            objective_value = \
                sklearn.metrics.accuracy_score(y_true=y_val, y_pred=y_pred) if self.task == 'classification' \
                else sklearn.metrics.mean_squared_error(y_true=y_val, y_pred=y_pred)
            # report value for pruning
            # step has an offset based on outerfold_number as same study is used for all outerfolds
            trial.report(value=objective_value, step=int(innerfold_name[-1]))
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            # store results and persist model
            objective_values.append(objective_value)
            validation_results[innerfold_name + '_val_true'] = y_val
            validation_results[innerfold_name + '_val_pred'] = y_pred
            validation_results.update(eval_metrics.get_evaluation_report(y_pred=y_pred, y_true=y_val, task=self.task,
                                                                         prefix=innerfold_name + '_'))
            model.save_model(path=self.save_path + 'temp/',
                             filename=innerfold_name + '-validation_model_trial' + str(trial.number))
        # persist results
        pd.DataFrame(columns=validation_results.keys()).append(validation_results, ignore_index=True).to_csv(
            self.save_path + 'temp/validation_results_trial' + str(trial.number) + '.csv')

        return np.mean(objective_values)

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
            # Start optimization run
            self.study.optimize(
                lambda trial: self.objective(trial=trial, train_val_indices=outerfold_info),
                n_trials=self.arguments.n_trials,
                n_jobs=-1
            )
            # Print statistics after run
            print("## Optuna Study finished ##")
            print("Study statistics: ")
            print("  Number of finished trials: ", len(self.study.trials))
            print("  Best Trial: ", self.study.best_trial.number)
            print("  Value: ", self.study.best_trial.value)
            print("  Params: ")
            for key, value in self.study.best_trial.params.items():
                print("    {}: {}".format(key, value))

            # Only keep validation results and models of best trial
            files_to_keep = glob.glob(self.save_path + 'temp/' + '*trial' + str(self.study.best_trial.number) + '*')
            for file in files_to_keep:
                shutil.copy2(file, self.save_path)
            shutil.rmtree(self.save_path + 'temp/')

            # Retrain on full train + val data with best hyperparams and apply on test
            X_test, y_test = self.dataset.X_full[outerfold_info['test']], self.dataset.y_full[outerfold_info['test']]
            X_retrain, y_retrain = \
                self.dataset.X_full[~outerfold_info['test']], self.dataset.y_full[~outerfold_info['test']]
            final_model = joblib.load(self.save_path + 'unfitted_model_trial' + str(self.study.best_trial.number))
            final_model.train(X_train=X_retrain, y_train=y_retrain)
            y_pred_retrain = final_model.predict(X_retrain)
            y_pred_test = final_model.predict(X_test)

            # Evaluate and save results
            eval_scores = eval_metrics.get_evaluation_report(y_pred=y_pred_test, y_true=y_test, task=self.task)
            print('## Results on test set ##')
            print(eval_scores)
            final_results = {'y_pred_test': y_pred_test, 'y_true_test': y_test,
                             'y_pred_retrain': y_pred_retrain, 'y_true_retrain': y_retrain}
            final_results.update(eval_scores)
            pd.DataFrame(columns=final_results.keys()).append(final_results, ignore_index=True).to_csv(
                self.save_path + 'final_model_test_results.csv')
            final_model.save_model(path=self.save_path,
                                   filename='final_retrained_model')
