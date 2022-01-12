import sys
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from math import floor, isnan
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import pdb
import os
import optuna
import datetime
from sklearn import metrics
import copy
import argparse
import glob


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_loadersH5(h5_path, outerfold_idx, innerfold_idx, b_onehot, batch_size, n_workers=0):
    train_ds = SNPDataset(h5_path=h5_path,
                          outerfold_idx=outerfold_idx,
                          innerfold_idx=innerfold_idx,
                          train_val_str="trn",
                          b_onehot=b_onehot)
    if train_ds.n_samples % 32 == 1:
        train_ds.y = train_ds.y[..., 0:-1]
        train_ds.X = train_ds.X[0:-1, ...]
        train_ds.sid = train_ds.sid[..., 0:-1]
        train_ds.n_samples -= 1
    train_loader = DataLoader(dataset=train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=n_workers,
                              pin_memory=True)

    val_ds = SNPDataset(h5_path=h5_path,
                        outerfold_idx=outerfold_idx,
                        innerfold_idx=innerfold_idx,
                        train_val_str="vld",
                        b_onehot=b_onehot)
    if val_ds.n_samples % 32 == 1:
        val_ds.y = val_ds.y[..., 0:-1]
        val_ds.X = val_ds.X[0:-1, ...]
        val_ds.sid = val_ds.sid[..., 0:-1]
        val_ds.n_samples -= 1

    val_loader = DataLoader(dataset=val_ds,
                            batch_size=batch_size,
                            num_workers=n_workers,
                            pin_memory=True)

    return train_loader, val_loader, val_ds.sid


class SNPDataset(Dataset):
    def __init__(self, h5_path, outerfold_idx, innerfold_idx, train_val_str, b_onehot):
        with h5py.File(h5_path, "r") as f:
            sid = f[f'outerfold_{outerfold_idx}/innerfold_{innerfold_idx}/{train_val_str}/sid'][:]
            X = f[f'outerfold_{outerfold_idx}/innerfold_{innerfold_idx}/{train_val_str}/X'][:]
            X_onehot = f[f'outerfold_{outerfold_idx}/innerfold_{innerfold_idx}/{train_val_str}/X_onehot'][:]
            y = f[f'outerfold_{outerfold_idx}/innerfold_{innerfold_idx}/{train_val_str}/y'][:]

        self.sid = torch.from_numpy(sid)
        self.y = torch.from_numpy(y)
        self.n_samples = len(sid)

        if b_onehot == True:
            self.X = torch.from_numpy(X_onehot)
        else:
            X = np.where(X < 1, 0, 1)
            self.X = torch.from_numpy(X)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_one_batch(inputs, targets, optimizer, model, loss_fn, scaler, device, method, l1_lambda=0,
                    l1_loss_added=False):
    optimizer.zero_grad()
    # forward with AMP
    with torch.cuda.amp.autocast():  # for AMP
        inputs = inputs.float().to(device=device)
        targets = targets.float().to(device=device)
        predictions = model(inputs)
        if method == "classification":
            # one_hot = torch.nn.functional.one_hot(targets.to(torch.int64), 2)
            # y_hot = one_hot.type_as(predictions)
            # loss = loss_fn(predictions, y_hot)
            loss = loss_fn(predictions, torch.reshape(targets, (-1, 1)))
        else:
            loss = loss_fn(predictions, torch.reshape(targets, (-1, 1)))
        if l1_loss_added:
            reg_loss = 0
            for param in model.parameters():
                reg_loss += param.norm(1)  # torch.sum(torch.abs(param))
            loss += l1_lambda * reg_loss

    scaler.scale(loss).backward()  # for AMP
    nn.utils.clip_grad_value_(model.parameters(), 0.1)  # for AMP
    scaler.step(optimizer)  # for AMP
    scaler.update()  # for AMP
    inputs = inputs.detach().cpu()
    targets = targets.detach().cpu()
    predictions = predictions.detach().cpu()

    # TODO: add sheduler step here, if for each batch
    return loss.item()


def train_one_epoch(data_loader, model, optimizer, loss_fn, scaler, epoch, device, method, l1_lambda=0, l1_loss_added=False):
    '''
    does 1 epoch of training
    '''
    model.train()
    epoch_loss = 0
    # loop = tqdm(data_loader)
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        batch_loss = train_one_batch(inputs=inputs,
                                     targets=targets,
                                     optimizer=optimizer,
                                     model=model,
                                     loss_fn=loss_fn,
                                     scaler=scaler,
                                     device=device,
                                     method=method,
                                     l1_loss_added=l1_loss_added,
                                     l1_lambda=l1_lambda)
        epoch_loss += batch_loss
        # loop.set_description(f"E{epoch+1:03d}")
        # loop.set_postfix_str(s=f"epoch_loss {epoch_loss:.3f}")

    return epoch_loss


def validate_one_batch(inputs, targets, model, loss_fn, device, method):
    with torch.cuda.amp.autocast():  # for AMP
        inputs = inputs.float().to(device=device)
        targets = targets.float().to(device=device)
        predictions = model(inputs)
        # pred = torch.round(torch.sigmoid(outputs))
        # correct += (pred.flatten() == targets).sum().item()
        if method == "classification":
            # one_hot = torch.nn.functional.one_hot(targets.to(torch.int64), 2)
            # y_hot = one_hot.type_as(predictions)
            # loss = loss_fn(predictions, y_hot)
            # accuracy as val loss -> calculate correct classifications
            predictions = torch.round(torch.sigmoid(predictions))
            loss = (predictions.flatten() == targets).sum().item()
        else:
            loss = loss_fn(predictions, torch.reshape(targets, (-1, 1)))
    inputs = inputs.detach().cpu()
    targets = targets.detach().cpu()
    predictions = predictions.detach().cpu()
    return loss, predictions


def validate_one_epoch(model, data_loader, loss_fn, epoch, device, method):
    model.eval()
    val_epoch_loss = 0
    preds = None
    for batch_index, (inputs, targets) in enumerate(data_loader):
        with torch.no_grad():
            loss, output = validate_one_batch(inputs, targets, model, loss_fn, device, method)
        val_epoch_loss += loss
        if preds is None:
            preds = torch.clone(output)
        else:
            preds = torch.cat((preds, output))
    if method == 'classification':
        # calculate accuracy based on correct classifications and amount of samples
        val_epoch_loss = val_epoch_loss / data_loader.dataset.n_samples

    return val_epoch_loss, preds


def define_model(trial, n_features, retraining=False, method="regression"):
    layers = []
    in_width = n_features
    in_channels = 4  # 4 for one_hot data

    if retraining == False:
        n_layers = trial.suggest_int("n_layers", 1, 3)
        out_channels = trial.suggest_int("out_channels", 1, 3)  # out channels for each conv layer
    else:
        n_layers = trial.params['n_layers']
        out_channels = trial.params['out_channels']

    def calc_conv_out(input_width, kernel_size, padding, stride):
        output_width = ((input_width - kernel_size + 2 * padding) / stride) + 1
        return floor(output_width)

    for i in range(n_layers):
        if retraining == False:
            out_channels = trial.suggest_int(f"out_channels_l{i}", 1, 8)
            dropout_percent = trial.suggest_float(f"dropout_l{i}", 0.2, 0.8)
            kernel_size_exp = trial.suggest_int("kernel_size_exp", 4, 7)
            kernel_size = 2 ** kernel_size_exp
            stride_exp = trial.suggest_int("stride_exp", 10, 12)
            stride = 2 ** stride_exp
        else:
            out_channels = trial.params[f"out_channels_l{i}"]
            dropout_percent = trial.params[f"dropout_l{i}"]
            kernel_size_exp = trial.params['kernel_size_exp']
            kernel_size = 2 ** kernel_size_exp
            stride_exp = trial.params['stride_exp']
            stride = 2 ** stride_exp

        # Conv1D could do padding same, but then we can only do stride=1
        layers.append(nn.Conv1d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride))

        out_width = calc_conv_out(input_width=in_width, kernel_size=kernel_size, padding=0, stride=stride)

        layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_percent, inplace=True))
        in_width = out_width
        in_channels = out_channels

    layers.append(nn.Flatten())
    layers.append(nn.Linear(out_channels * out_width, 1024))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(1024, 512))
    layers.append(nn.ReLU())
    if method == "classification":
        layers.append(nn.Linear(512, 1))  # return logits # layers.append(nn.Linear(512, 2))
    else:
        layers.append(nn.Linear(512, 1))

    return nn.Sequential(*layers)


def suggestLearningHyperparameterRanges(trial, model):
    lr = trial.suggest_loguniform("lr", 1e-6, 1e-2)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    return optimizer


def restart_from_checkpoint(study, trial):
    # Enqueue trial with the same parameters as the stale trial to use saved information.

    path = f"./checkpoints/cnn_class/{trial.number}/model.pt"
    user_attrs = copy.deepcopy(trial.user_attrs)
    if os.path.exists(path):
        user_attrs["checkpoint_path"] = path

    study.add_trial(
        optuna.create_trial(
            state=optuna.trial.TrialState.WAITING,
            params=trial.params,
            distributions=trial.distributions,
            user_attrs=user_attrs,
            system_attrs=trial.system_attrs,
        )
    )


def objective(trial, in_feature_shape, h5_path, save_path, device, num_folds=3, num_epochs=5, method="regression",
              l1_loss_added=False):
    print(f'Starting Trial {trial.number}, Method: {method}')
    outerfold_idx = 0  # we train only on 1. outerfold, as there is no time to do all folds
    save_path = f"{save_path}/{trial.number}"
    os.makedirs(save_path, exist_ok=True)

    # for mse loss we need to start with high numbers, so we can save the better val loss per current epoch
    if method == "regression":
        loss_total = loss_total = np.ones(num_folds) * 99999999
        loss_fn = nn.MSELoss()
        mode = 'min'
    else:
        loss_total = np.zeros(num_folds)
        loss_fn = nn.BCEWithLogitsLoss()
        mode = 'max'
    if l1_loss_added:
        l1_lambda = trial.suggest_float("l1_lambda", 1e-10, 1)
    else:
        l1_lambda = 0
    try:
        model = define_model(trial, n_features=in_feature_shape[1], method=method).to(device)
    except Exception as exc:
        print(exc)
        print("Failed to init model. Trial broken.")
        return np.mean(loss_total)
    print(trial.params)
    print(model)
    optimizer = suggestLearningHyperparameterRanges(trial, model)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=0.1, min_lr=1e-5, patience=3)
    scaler = torch.cuda.amp.GradScaler()
    early_stopping_threshold = 0.1 * num_epochs
    # Load checkpoint if already exists. This is for unexpected cancellation of a study, so we don't need to start fresh
    if "checkpoint_path" in trial.user_attrs:
        checkpoint = torch.load(trial.user_attrs["checkpoint_path"])
        epoch_begin = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        train_loss = checkpoint["train_loss"]
        val_loss = checkpoint["valid_loss"]
        # acc = checkpoint["acc"]
        # loss_total = checkpoint["loss_total"]
        outerfold_idx = checkpoint["outerfold_idx"]
        innerfold_idx = checkpoint["innerfold_idx"]
        print("Existing Model loaded.")
    else:
        epoch_begin = 0

    for innerfold_idx in range(num_folds):
        print('Fold ' + str(innerfold_idx))
        # Get the dataloaders
        train_loader, valid_loader, _ = get_loadersH5(h5_path=h5_path,
                                                      outerfold_idx=outerfold_idx,
                                                      innerfold_idx=innerfold_idx,
                                                      b_onehot=True,
                                                      batch_size=32)
        epochs_no_improve = 0
        for epoch in range(epoch_begin, num_epochs):
            train_loss = train_one_epoch(data_loader=train_loader,
                                         model=model,
                                         optimizer=optimizer,
                                         loss_fn=loss_fn,
                                         scaler=scaler,
                                         epoch=epoch,
                                         device=device,
                                         method=method,
                                         l1_loss_added=l1_loss_added,
                                         l1_lambda=l1_lambda)

            val_loss, preds = validate_one_epoch(model=model,
                                                 data_loader=valid_loader,
                                                 loss_fn=loss_fn,
                                                 epoch=epoch,
                                                 device=device,
                                                 method=method)
            scheduler.step(val_loss)
            valid_loss = val_loss.item() if method == 'regression' else val_loss
            print('loss:' + str(valid_loss))
            # save val_loss only, if its better than the current
            if (method == 'regression' and valid_loss < loss_total[innerfold_idx]) or \
                    (method == 'classification' and valid_loss > loss_total[innerfold_idx]):
                loss_total[innerfold_idx] = valid_loss
                print("new best")
                epochs_no_improve = 0
                results = pd.DataFrame(columns=['True', 'Prediction'])
                results['True'] = valid_loader.dataset.y
                results['Prediction'] = preds.numpy().flatten().astype(float)
                # outer_fold_number = data_path.split('/')[-2][-1]
                # X_loaded = np.loadtxt(data_path + 'test_X_' + outer_fold_number + '_full.csv', delimiter=",")
                results.index = valid_loader.dataset.sid
                path_to_save = \
                    h5_path[:h5_path.rfind('/')] + '/outerFold' + str(outerfold_idx) + '/predictions_cnn/'
                if not os.path.exists(path_to_save):
                    os.makedirs(path_to_save)
                results.to_csv(
                    path_to_save + 'Trial' + str(trial.number) + '_' + 'Predictions_Validation_'
                    + str(outerfold_idx) + '_' + str(innerfold_idx) + '_'
                    + study.study_name + '.csv',
                    sep=',', decimal='.', float_format='%.10f')
            else:
                epochs_no_improve += 1
            # report actual val_loss, so the optimize function will work
            if not isnan(valid_loss):
                trial.report(valid_loss, epoch)

            # Save optimization status. We should save the objective value because the process may be
            # killed between saving the last model and recording the objective value to the storage.
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "valid_loss": val_loss,
                    "outerfold_idx": outerfold_idx,
                    "innerfold_idx": innerfold_idx
                },
                os.path.join(save_path, "model.pt"),
            )

            # Handle pruning based on the intermediate value.
            # model that are starting bad will be stopped earlier, so we don't waste time running them
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            if epochs_no_improve >= early_stopping_threshold:
                print("Early Stopping")
                break
    print("Result over all folds:" + str(np.mean(loss_total)))
    return np.mean(loss_total)


def train_final_model(data_path, best_trial, device, n_epochs, outer_fold_number, method, n_feature_shape,
                      l1_loss_added=False, pca=False):
    if method == "classification":
        loss_fn = nn.BCEWithLogitsLoss()
    elif method == "regression":
        loss_fn = nn.MSELoss()
    else:
        print("invalid method.")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    model = define_model(best_trial, n_features=in_feature_shape[1], retraining=True, method=method).to(device)
    print(model)
    # Generate the optimizers.
    lr = best_trial.params['lr']
    optimizer = getattr(optim, best_trial.params['optimizer'])(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader, test_indices = get_loadersH5(h5_path=data_path,
                                                            outerfold_idx=outer_fold_number,
                                                            innerfold_idx="full",
                                                            b_onehot=True,
                                                            batch_size=32)
    if l1_loss_added:
        l1_lambda = best_trial.params['l1_lambda']
    else:
        l1_lambda = 0

    for epoch in range(n_epochs):
        train_loss = train_one_epoch(data_loader=train_loader,
                                     model=model,
                                     optimizer=optimizer,
                                     loss_fn=loss_fn,
                                     scaler=scaler,
                                     epoch=epoch,
                                     device=device,
                                     method=method,
                                     l1_loss_added=l1_loss_added,
                                     l1_lambda=l1_lambda)

        # Testing
        model.eval()
        correct = 0
        targets = None
        preds = None
        for batch_index, (inputs, target) in enumerate(test_loader):
            with torch.no_grad():
                inputs = inputs.float().to(device=device)
                target = target.float()
                outputs = model(inputs)
                outputs = outputs.detach().cpu()

                if method == "classification":
                    pred = torch.round(torch.sigmoid(outputs))  # pred = torch.argmax(outputs, dim=1)
                    correct += (pred.flatten() == target).sum().item()
                else:
                    pred = outputs

                if preds is None:
                    preds = torch.clone(pred)
                    targets = torch.clone(torch.reshape(target, (-1, 1)))
                else:
                    preds = torch.cat((preds, pred))
                    targets = torch.cat((targets, torch.reshape(target, (-1, 1))))

    # check results
    print('***** Results on Test Set ******')
    if method == "classification":
        acc = correct / test_loader.dataset.n_samples
        print('Acc = ' + str(acc))
    else:
        mse = loss_fn(preds, targets).item()
        print('MSE = ' + str(mse))
        print('RMSE = ' + str(np.sqrt(mse)))
    # save to csv
    results = pd.DataFrame(columns=['True', 'Prediction'])
    # pdb.set_trace()
    # print(targets.shape)
    results['True'] = targets.numpy().flatten().astype(float)
    results['Prediction'] = preds.numpy().flatten().astype(float)

    results.index = test_indices.numpy()  # X_loaded[:, 0].astype(int)
    path_to_save = \
        data_path[:data_path.rfind('/')] + '/outerFold' + str(outer_fold_number) + '/predictions_cnn/'
    print(f"Save Path: {path_to_save}")
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    results.to_csv(
        path_to_save + 'Predictions_Test_' + str(outer_fold_number) + '_' + 'full' + '_'
        + study.study_name + '-' + datetime.datetime.now().strftime("%d-%b-%Y_%H-%M") + '.csv',
        sep=',', decimal='.', float_format='%.10f')
    torch.save(model.state_dict(), path_to_save + 'Checkpoints_' + study.study_name
               + '-' + datetime.datetime.now().strftime("%d-%b-%Y_%H-%M") + '.pt')


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("-trials", "--n_trials", type=int, default=100,
                        help="number of trials for optuna")
    parser.add_argument("-baseset", "--base_dataset", type=str, default="atwell_FT10_s",
                        help="base dataset to use")
    parser.add_argument("-kfolds", "--kfolds", type=int, default=3,
                        help="number of folds inner loop")
    parser.add_argument("-nepochs", "--nepochs", type=int, default=500,
                        help="number of epochs")
    parser.add_argument("-l1", '--l1_loss', type=bool, default=False)
    # parser.add_argument("-method", '--method', type=str, default='regression', help="regression or classification")

    args = parser.parse_args()
    N_TRIALS = args.n_trials
    baseset = args.base_dataset
    NUM_FOLDS = args.kfolds
    NUM_EPOCHS = args.nepochs
    l1_loss_added = args.l1_loss
    # METHOD = args.method

    BASE_PATH = '/bit_storage/Workspace/Maura/ML4RG/'
    DB_PATH = "./optuna_db"
    SAVE_PATH = f"./checkpoints/cnn_class/"

    SEED = 42
    if 'FT10' in baseset:
        METHOD = "regression"
        STUDY_NAME = "CNN-regression_trials-" + str(N_TRIALS) + '_epochs-' + str(NUM_EPOCHS) + '_kfolds-' + str(
            NUM_FOLDS) + '_l1-' + str(l1_loss_added) + '_' + baseset
        direction = 'minimize'
    else:
        METHOD = "classification"
        STUDY_NAME = "CNN-classification_trials-" + str(N_TRIALS) + '_epochs-' + str(NUM_EPOCHS) + '_kfolds-' + str(
            NUM_FOLDS) + '_l1-' + str(l1_loss_added) + '_' + baseset
        direction = 'maximize'

    if 'copy' not in baseset:
        h5_path = BASE_PATH + 'train_test_split/' + baseset + '/' + baseset + '_strat.h5'
    else:
        h5_path = BASE_PATH + 'train_test_split/' + baseset.split('-')[1] + '/' + baseset + '_strat.h5'
    DB_NAME = "ml4rg_" + STUDY_NAME

    outerfold_idx = 0
    innerfold_idx = 1
    valid_sid_ls = []
    with h5py.File(h5_path, "r") as f:
        valid_sid = f[f'outerfold_{outerfold_idx}/innerfold_{innerfold_idx}/trn/sid'][:]
        valid_X = f[f'outerfold_{outerfold_idx}/innerfold_{innerfold_idx}/vld/X'][:]
        valid_X_onehot = f[f'outerfold_{outerfold_idx}/innerfold_{innerfold_idx}/vld/X_onehot'][:]
        valid_y = f[f'outerfold_{outerfold_idx}/innerfold_{innerfold_idx}/vld/y'][:]
        valid_sid_ls.append(valid_sid)

    in_feature_shape = valid_X_onehot.shape[1:]

    seed_all(SEED)
    os.makedirs(DB_PATH, exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=SEED)
    storage = optuna.storages.RDBStorage(f"sqlite:///{DB_PATH}/{DB_NAME}.db", heartbeat_interval=1,
                                         failed_trial_callback=restart_from_checkpoint)
    study = optuna.create_study(storage=storage, study_name=STUDY_NAME, sampler=sampler, direction=direction,
                                load_if_exists=True)

    study.optimize(
        lambda trial: objective(trial, h5_path=h5_path, save_path=SAVE_PATH, num_epochs=NUM_EPOCHS, num_folds=NUM_FOLDS,
                                in_feature_shape=in_feature_shape, method=METHOD, device=device,
                                l1_loss_added=l1_loss_added), n_trials=N_TRIALS)

    best_trial = study.best_trial
    print('Best Trial')
    print(best_trial)

    # delete val files of not best studies
    path_to_save = \
        h5_path[:h5_path.rfind('/')] + '/outerFold' + str(outerfold_idx) + '/predictions_cnn/'
    all_files = glob.glob(path_to_save + '*' + STUDY_NAME + '.csv')
    all_files = [file for file in all_files if 'Trial' + str(study.best_trial.number) not in file]
    for f in all_files:
        os.remove(f)

    train_final_model(data_path=h5_path,
                      best_trial=best_trial,
                      device=device,
                      n_epochs=NUM_EPOCHS,
                      outer_fold_number=0,
                      n_feature_shape=in_feature_shape,
                      l1_loss_added=l1_loss_added,
                      pca=False,
                      method=METHOD)

