import numpy as np
import joblib
import tensorflow as tf
import pathlib
import pandas as pd
import optuna

from . import _base_model, _tensorflow_model, _param_free_base_model, _torch_model
from ..evaluation import eval_metrics
from ..utils import helper_functions
from ..postprocess import results_analysis
from ..preprocess import base_dataset


def load_retrain_model(path: pathlib.Path, filename: str, X_retrain: np.array, y_retrain: np.array,
                       early_stopping_point: int = None) -> _base_model.BaseModel:
    """
    Load and retrain persisted model

    :param path: path where the model is saved
    :param filename: filename of the model
    :param X_retrain: feature matrix for retraining
    :param y_retrain: target vector for retraining
    :param early_stopping_point: optional early stopping point relevant for some models

    :return: model instance
    """
    model = load_model(path=path, filename=filename)
    if early_stopping_point is not None:
        model.early_stopping_point = early_stopping_point
    model.retrain(X_retrain=X_retrain, y_retrain=y_retrain)
    return model


def retrain_model_with_results_file(results_file_path: pathlib.Path, model_name: str, datasplit: str,
                                    outerfold_number: int, dataset: base_dataset.Dataset,
                                    saved_outerfold_number: int = None, saved_datasplit: str = None) \
        -> _base_model.BaseModel:
    """
    Retrain a model based on information saved in a results overview file.

    :param results_file_path: path to the results overview file containing the information for retraining
    :param model_name: name of the model to retrain
    :param datasplit: datasplit to use
    :param outerfold_number: outerfold number to use
    :param dataset: dataset for training
    :param saved_outerfold_number: outerfold number of the results file in case the model is retrained with fixed hyperparameters of one outerfold
    :param saved_datasplit: outerfold number of the results file in case the model is retrained on another datasplit

    :return: retrained model
    """

    saved_outerfold_number = outerfold_number if saved_outerfold_number is None else saved_outerfold_number
    saved_datasplit = datasplit if saved_datasplit is None else saved_datasplit
    outerfold_info = dataset.datasplit_indices['outerfold_' + str(outerfold_number)]
    X_test, y_test, sample_ids_test = \
        dataset.X_full[outerfold_info['test']], dataset.y_full[outerfold_info['test']], \
        dataset.sample_ids_full[outerfold_info['test']]
    X_retrain, y_retrain, sample_ids_retrain = \
        dataset.X_full[~np.isin(np.arange(len(dataset.X_full)), outerfold_info['test'])], \
        dataset.y_full[~np.isin(np.arange(len(dataset.y_full)), outerfold_info['test'])], \
        dataset.sample_ids_full[~np.isin(np.arange(len(dataset.sample_ids_full)), outerfold_info['test'])]

    results = pd.read_csv(results_file_path)
    results = results[results[results.columns[0]] == 'outerfold_' + str(saved_outerfold_number)] \
        if saved_datasplit == 'nested-cv' else results
    results = results.loc[:, [model_name in col for col in results.columns]]
    eval_dict_saved = results_analysis.result_string_to_dictionary(
        result_string=results[model_name + '___eval_metrics'][saved_outerfold_number]
    )
    task = 'regression' if 'test_rmse' in eval_dict_saved.keys() else 'classification'
    helper_functions.set_all_seeds()
    if issubclass(helper_functions.get_mapping_name_to_class()[model_name],
                  _param_free_base_model.ParamFreeBaseModel):
        model: _param_free_base_model.ParamFreeBaseModel = \
            helper_functions.get_mapping_name_to_class()[model_name](
                task=task,
            )
        _ = model.fit(X=X_retrain, y=y_retrain)
    else:
        best_params = results_analysis.result_string_to_dictionary(
            result_string=results[model_name + '___best_params'][saved_outerfold_number]
        )
        trial = optuna.trial.FixedTrial(params=best_params)
        additional_attributes_dict = {}
        if issubclass(helper_functions.get_mapping_name_to_class()[model_name], _torch_model.TorchModel) or \
                issubclass(helper_functions.get_mapping_name_to_class()[model_name], _tensorflow_model.TensorflowModel):
            # additional attributes for torch and tensorflow models
            additional_attributes_dict['n_features'] = dataset.X_full.shape[1]
            additional_attributes_dict['width_onehot'] = dataset.X_full.shape[-1]
            additional_attributes_dict['early_stopping_point'] = best_params['early_stopping_point']
            additional_attributes_dict['n_epochs'] = best_params['n_epochs']
            additional_attributes_dict['batch_size'] = best_params['batch_size']
        model: _base_model.BaseModel = helper_functions.get_mapping_name_to_class()[model_name](
            task=task, optuna_trial=trial,
            n_outputs=len(np.unique(dataset.y_full)) if task == 'classification' else 1,
            **additional_attributes_dict
        )
        model.retrain(X_retrain=X_retrain, y_retrain=y_retrain)
    y_pred_test = model.predict(X_in=X_test)
    eval_scores = \
        eval_metrics.get_evaluation_report(y_pred=y_pred_test, y_true=y_test,
                                           task=model.task,
                                           prefix='test_')
    print('Compare Results from initial testing to refitting')
    print('New fitting: ')
    print(eval_scores)
    print('Old fitting: ')
    print(eval_dict_saved)

    return model


def load_model(path: pathlib.Path, filename: str) -> _base_model.BaseModel:
    """
    Load persisted model

    :param path: path where the model is saved
    :param filename: filename of the model

    :return: model instance
    """
    model = joblib.load(path.joinpath(filename))
    # special case for loading tensorflow optimizer
    if issubclass(type(model), _tensorflow_model.TensorflowModel):
        model.optimizer = tf.keras.optimizers.deserialize(model.optimizer)
    return model

