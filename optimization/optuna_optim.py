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
    """Class that contains all info for the whole optimization using optuna for one model and dataset"""

    def __init__(self, arguments: argparse.Namespace, task: str, current_model_name: str,
                 dataset: base_dataset.Dataset):
        self.current_model_name = current_model_name
        self.task = task
        self.study_name = str(datetime.datetime.now()) + '_' + \
            arguments.genotype_matrix + '-' + arguments.phenotype_matrix + '-' + arguments.phenotype + '-' + \
            'MAF' + str(arguments.maf_percentage) + \
            '-SPLIT' + arguments.datasplit + helper_functions.get_subpath_for_datasplit(arguments=arguments) + \
            '-MODEL' + arguments.model + '-TRIALS' + str(arguments.n_trials)
        self.save_path = arguments.base_dir + 'results/' + \
            arguments.genotype_matrix + '/' + arguments.phenotype_matrix + '/' + arguments.phenotype + \
            '/' + current_model_name + '/' + helper_functions.get_subpath_for_datasplit(arguments=arguments) + '/' + \
            str(datetime.datetime.now()) + '/'
        storage = optuna.storages.RDBStorage(
            "sqlite:////" + self.save_path + self.study_name + ".db", heartbeat_interval=10
            )
        self.study = optuna.create_study(
            storage=storage, study_name=self.study_name,
            direction='minimize' if task == 'regression' else 'maximize', load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.PercentilePruner(percentile=80, n_min_trials=20)
        )
        self.dataset = dataset

    def objective(self, trial: optuna.trial.Trial, train_val_indices: dict, outerfold_number: int):
        # Create model
        # in case a model has attributes not part of the base class hand them over in a dictionary to keep the same call
        # (name of the attribute and key in the dictionary have to match)
        additional_attributes_dict = {}
        model = utils.helper_functions.get_mapping_name_to_class()[self.current_model](
            task=self.task, optuna_trial=trial, **additional_attributes_dict
        )

        # Iterate over all innerfolds
        objective_values = []
        validation_results = {}
        os.makedirs(self.save_path + 'temp/')
        for innerfold_name, innerfold_info in train_val_indices.items():
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
            model.reset_model()
            model.train(X_train=X_train, y_train=y_train)
            # validate model
            y_pred = model.predict(X_in=X_val)
            objective_value = \
                sklearn.metrics.accuracy_score(y_true=y_val, y_pred=y_pred) if self.task == 'classification' \
                else sklearn.metrics.mean_squared_error(y_true=y_val, y_pred=y_pred)
            # report value for pruning
            trial.report(value=objective_value, step=outerfold_number * 100 + int(innerfold_name[-1]))
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

    def run_optuna_optimization(self, arguments: argparse.Namespace):

        # Iterate over outerfolds
        # (according to structure described in base_dataset.Dataset, only for nested-cv multiple outerfolds exist)
        for outerfold_name, outerfold_info in self.dataset.datasplit_indices.items():
            if self.dataset.datasplit == 'nested-cv':
                # Only print outerfold info for nested-cv as it does not apply for the other splits
                print("## Starting Optimization for " + outerfold_name + " ##")
                self.save_path += outerfold_name + '/'
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
            # Start optimization run
            self.study.optimize(
                lambda trial: self.objective(trial=trial, train_val_indices=outerfold_info,
                                             outerfold_number=int(outerfold_name[-1])),
                n_trials=arguments.n_trials,
                n_jobs=-1
            )
            # Print statistics after run
            print("## Optuna Study finished ##")
            print("Study statistics: ")
            print("  Number of finished trials: ", len(self.study.trials))
            print("  Number of pruned trials: ", len(self.pruned_trials))
            print("  Number of complete trials: ", len(self.complete_trials))

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
            # TODO: indexing testen
            X_test, y_test = self.dataset.X_full[outerfold_info['test']], self.dataset.y_full[outerfold_info['test']]
            X_retrain, y_retrain = \
                self.dataset.X_full[~outerfold_info['test']], self.dataset.y_full[~outerfold_info['test']]
            final_model = joblib.load([file for file in files_to_keep if 'model' in file][0])
            final_model.reset_loaded_model()
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
