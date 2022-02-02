import argparse
import datetime
import optuna
import utils


class OptunaOptim:
    """Class that contains all info for the whole optimization using optuna for one model and dataset"""

    def __init__(self, arguments: argparse.Namespace, task: str, current_model_name: str):
        self.current_model_name = current_model_name
        self.task = task
        self.study_name = str(datetime.datetime.now()) + '_' + \
            arguments.genotype_matrix + '-' + arguments.phenotype_matrix + '-' + arguments.phenotype + '-' + \
            'MAF' + str(arguments.maf_percentage) + '-SPLIT' + arguments.datasplit + '-MODEL' + arguments.model + \
            '-TRIALS' + str(arguments.n_trials)
        self.save_path = arguments.base_dir + 'results/' + \
            arguments.genotype_matrix + '/' + arguments.phenotype_matrix + '/' + arguments.phenotype + \
            '/' + current_model_name + '/'
        storage = optuna.storages.RDBStorage(
            "sqlite:////" + self.save_path + self.study_name + ".db", heartbeat_interval=10
            )
        self.study = optuna.create_study(
            storage=storage, study_name=self.study_name,
            direction='minimize' if task == 'regression' else 'maximize', load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42)
        )

    def objective(self, trial: optuna.trial.Trial):
        # Modell instantiieren mit übergebenem Trial
        # Model train etc. aufrufen

        model = utils.helper_functions.get_mapping_name_to_class()[self.current_model](
            task=self.task, optuna_trial=trial
        )

        # iterieren über folds
        # modell aufrufen und einen trainingslauf durchführen
        # mean bilden und zurückgeben
        # modelle + ergebnisse abspeichern --> solche die nicht der beste run waren wieder löschen vom val
        return 0

    def run_optuna_optimization(self, arguments: argparse.Namespace):

        # über outer folds iterieren
        # Start optimization run
        self.study.optimize(
            lambda trial: self.objective(trial),
            n_trials=arguments.n_trials,
            n_jobs=-1
        )

        # Print statistics after run
        print("### Optuna Study finished ###")
        print("Study statistics: ")
        print("  Number of finished trials: ", len(self.study.trials))
        print("  Number of pruned trials: ", len(self.pruned_trials))
        print("  Number of complete trials: ", len(self.complete_trials))

        print("Best trial:")
        trial = self.study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        # TODO: validierungsergebnisse wieder löschen, die nicht best trial sind
        # TODO: retrain und test - Ergebnisse + finales Modell abspeichern
