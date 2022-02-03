import copy
import os
import numpy as np
import math
import argparse
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
import datetime
import glob
import sklearn
import datetime

from SNPDataset import SNPDataset, SNPDatasetH5


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("[ Using Seed : ", seed, " ]")


def define_model(trial, n_features, pca=False):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    # ggfs. 1. Layer separat suggesten
    layers = []
    # X_loaded = np.loadtxt(data_path + 'train_X_0_0.csv', delimiter=",")
    # in_features = X_loaded.shape[1] - 1
    in_features = n_features
    if pca:
        out_features_first_layer_exp = trial.suggest_int("out_features_first_layer_exp", 3, 7)
    else:
        out_features_first_layer_exp = trial.suggest_int("out_features_first_layer_exp", 8, 10)
    out_features = 2 ** out_features_first_layer_exp
    layers.append(nn.Linear(in_features, out_features))
    layers.append(nn.ReLU())
    p_first_layer = trial.suggest_float("dropout_first_layer", 0.2, 0.7)
    layers.append(nn.Dropout(p_first_layer))
    in_features = out_features
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 5, 50)
        # out_features = 2 ** out_features_exp
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.7)
        layers.append(nn.Dropout(p))
        in_features = out_features
    layers.append(nn.Linear(in_features, 1))  # returns logits
    # layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)


def get_loaders(k, batch_size, data_path, n_workers=0):
    train_ds = SNPDataset(train_test='train', k=k, data_path=data_path)
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=n_workers, pin_memory=True)
    val_ds = SNPDataset(train_test='test', k=k, data_path=data_path)
    val_loader = DataLoader(dataset=val_ds, batch_size=batch_size, num_workers=n_workers, pin_memory=True)
    return train_loader, val_loader


def get_loadersH5(h5_path, outerfold_idx, innerfold_idx, batch_size, b_onehot=False, n_workers=0, pca=False):
    train_ds = SNPDatasetH5(h5_path=h5_path,
                            outerfold_idx=outerfold_idx,
                            innerfold_idx=innerfold_idx,
                            train_val_str="trn",
                            b_onehot=b_onehot)
    if pca:
        pca_transformer = sklearn.decomposition.PCA(0.95)
        train_ds.X = pca_transformer.fit_transform(train_ds.X)

    train_loader = DataLoader(dataset=train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=n_workers,
                              pin_memory=True)

    val_ds = SNPDatasetH5(h5_path=h5_path,
                          outerfold_idx=outerfold_idx,
                          innerfold_idx=innerfold_idx,
                          train_val_str="vld",
                          b_onehot=b_onehot)
    if pca:
        val_ds.X = pca_transformer.transform(val_ds.X)

    val_loader = DataLoader(dataset=val_ds,
                            batch_size=batch_size,
                            num_workers=n_workers,
                            pin_memory=True)

    return train_loader, val_loader, val_ds.sid


def objective(trial, device, data_path, outer_fold_number, k_folds=3, n_epochs=250, l1_loss_added=False, pca=False):
    # Generate the model
    ds_X = SNPDatasetH5(h5_path=data_path, outerfold_idx=outer_fold_number, innerfold_idx=0, train_val_str="trn").X
    if pca:
        pca_transformer = sklearn.decomposition.PCA(0.95)
        ds_X = pca_transformer.fit_transform(ds_X)
    n_features = ds_X.shape[1]
    model = define_model(trial, n_features=n_features, pca=pca).to(device)
    loss_fn = nn.BCEWithLogitsLoss()  # nn.MSELoss()

    # n_epochs = trial.suggest_int("epochs", 10, 50)
    # batch_size_exp = trial.suggest_int("batch_size_exp", 3, 6)
    # batch_size = 2 ** batch_size_exp
    batch_size = 32
    early_stopping_threshold = 0.1 * n_epochs

    # Generate the optimizers.
    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    if l1_loss_added:
        l1_lambda = trial.suggest_float("l1_lambda", 1e-10, 1)
    loss_total = np.zeros(k_folds)

    if "checkpoint_path" in trial.user_attrs:
        checkpoint = torch.load(trial.user_attrs["checkpoint_path"])
        epoch_begin = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        acc = checkpoint["acc"]
        loss_total = checkpoint["loss_total"]
    else:
        epoch_begin = 0

    path = f"mlp_class_pytorch_checkpoint/{trial.number}"
    os.makedirs(path, exist_ok=True)

    print('### Trial Params ###')
    print(model)
    print(trial.params)

    for k in range(k_folds):
        print('Fold ' + str(k))
        # Get the dataset.
        # train_loader, valid_loader = get_loaders(k=k, batch_size=batch_size, data_path=data_path)
        train_loader, valid_loader, valid_indices = get_loadersH5(h5_path=data_path, outerfold_idx=outer_fold_number,
                                                                  innerfold_idx=k, batch_size=batch_size, pca=pca)
        model[0] = nn.Linear(train_loader.dataset.X.shape[1], model[0].out_features)
        model = model.to(device)
        epochs_no_improve = 0
        # Training of the model.
        for epoch in range(epoch_begin, n_epochs):
            print('Epoch ' + str(epoch + 1) + '/' + str(n_epochs))
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.view(data.size(0), -1).to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data.float())
                loss = loss_fn(output, torch.reshape(target, (-1, 1)).float())
                if l1_loss_added:
                    reg_loss = 0
                    for param in model.parameters():
                        reg_loss += param.norm(1) #torch.sum(torch.abs(param))
                    loss += l1_lambda * reg_loss
                loss.backward()
                optimizer.step()
                data = data.detach().cpu()
                target = target.detach().cpu()
                output = output.detach().cpu()

            # Validation of the model.
            model.eval()
            correct = 0
            preds = None
            targets = None
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(valid_loader):
                    data, target = data.view(data.size(0), -1).to(device), target.to(device)
                    output = model(data.float())
                    data = data.detach().cpu()
                    target = target.detach().cpu()
                    output = output.detach().cpu()
                    # Get the index of the max log-probability.
                    pred = torch.round(torch.sigmoid(output))
                    correct += (pred.flatten() == target).sum().item()
                    if preds is None:
                        preds = torch.clone(pred)
                        targets = torch.clone(torch.reshape(target, (-1, 1)).float())
                    else:
                        preds = torch.cat((preds, pred))
                        targets = torch.cat((targets, torch.reshape(target, (-1, 1)).float()))
                #reg_loss = 0
                #for param in model.parameters():
                #    reg_loss += param.norm(1) # torch.sum(torch.abs(param))
                acc = correct / valid_loader.dataset.n_samples
                loss = acc  # + l1_lambda * reg_loss.item()
                print('acc: ' + str(acc))
                print('Loss: ' + str(loss))
                if acc > loss_total[k]:
                    print('new best')
                    loss_total[k] = acc
                    epochs_no_improve = 0
                    results = pd.DataFrame(columns=['True', 'Prediction'])
                    results['True'] = targets.numpy().flatten().astype(float)
                    results['Prediction'] = preds.numpy().flatten().astype(float)
                    # outer_fold_number = data_path.split('/')[-2][-1]
                    # X_loaded = np.loadtxt(data_path + 'test_X_' + outer_fold_number + '_full.csv', delimiter=",")
                    results.index = valid_indices  # X_loaded[:, 0].astype(int)
                    path_to_save = \
                        data_path[:data_path.rfind('/')] + '/outerFold' + str(outer_fold_number) + '/predictions/'
                    if not os.path.exists(path_to_save):
                        os.makedirs(path_to_save)
                    results.to_csv(
                        path_to_save + 'Trial' + str(trial.number) + '_' + 'Predictions_Validation_' + str(outer_fold_number) + '_' + str(k) + '_'
                        + study.study_name + '.csv',
                        sep=',', decimal='.', float_format='%.10f')
                else:
                    epochs_no_improve += 1
                if not math.isnan(acc):
                    trial.report(acc, epoch)

                # Save optimization status. We should save the objective value because the process may be
                # killed between saving the last model and recording the objective value to the storage.
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "acc": acc,
                        "loss_total": loss_total
                    },
                    os.path.join(path, "model.pt"),
                )

                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                if epochs_no_improve >= early_stopping_threshold:
                    print("Early Stopping")
                    break
    print("Result over all folds:" + str(np.mean(loss_total)))
    return np.mean(loss_total)


def restart_from_checkpoint(study, trial):
    # Enqueue trial with the same parameters as the stale trial to use saved information.

    path = "mlp_class_pytorch_checkpoint/{trial.number}/model.pt"
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


def train_final_model(data_path, best_trial, device, n_epochs, outer_fold_number, l1_loss_added=False, pca=False):
    ds_X = SNPDatasetH5(h5_path=data_path, outerfold_idx=outer_fold_number, innerfold_idx='full', train_val_str="trn").X
    if pca:
        pca_transformer = sklearn.decomposition.PCA(0.95)
        ds_X = pca_transformer.fit_transform(ds_X)
    n_features = ds_X.shape[1]
    layers = []
    in_features = n_features
    out_features = 2 ** best_trial.params['out_features_first_layer_exp']
    layers.append(nn.Linear(in_features, out_features))
    layers.append(nn.ReLU())
    p_first_layer = best_trial.params['dropout_first_layer']
    layers.append(nn.Dropout(p_first_layer))
    in_features = out_features
    for i in range(best_trial.params['n_layers']):
        out_features = best_trial.params["n_units_l{}".format(i)]
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = best_trial.params["dropout_l{}".format(i)]
        layers.append(nn.Dropout(p))
        in_features = out_features
    layers.append(nn.Linear(in_features, 1))

    model = nn.Sequential(*layers).to(device)
    loss_fn = nn.MSELoss()

    batch_size = 32  # 2 ** best_trial.params['batch_size_exp']

    # Generate the optimizers.
    lr = best_trial.params['lr']
    optimizer = getattr(optim, best_trial.params['optimizer'])(model.parameters(), lr=lr)

    if l1_loss_added:
        l1_lambda = best_trial.params['l1_lambda']
    # train_loader, test_loader = get_loaders(k='full', batch_size=batch_size, data_path=data_path)
    train_loader, test_loader, test_indices = get_loadersH5(h5_path=data_path, outerfold_idx=outer_fold_number,
                                                            innerfold_idx='full', batch_size=batch_size, pca=pca)
    # Training of the model.
    for epoch in range(n_epochs):
        print('Epoch ' + str(epoch + 1) + '/' + str(n_epochs))
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.view(data.size(0), -1).to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data.float())
            loss = loss_fn(output, torch.reshape(target, (-1, 1)).float())
            if l1_loss_added:
                reg_loss = 0
                for param in model.parameters():
                    reg_loss += param.norm(1)  # torch.sum(torch.abs(param))
                loss += l1_lambda * reg_loss
            loss.backward()
            optimizer.step()
            data = data.detach().cpu()
            target = target.detach().cpu()
            output = output.detach().cpu()

    # Predict on test set
    model.eval()
    correct = 0
    targets = None
    preds = None
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.view(data.size(0), -1).to(device), target.to(device)
            output = model(data.float())
            data = data.detach().cpu()
            target = target.detach().cpu()
            output = output.detach().cpu()
            pred = torch.round(torch.sigmoid(output))
            correct += (pred.flatten() == target).sum().item()
            if preds is None:
                preds = torch.clone(pred)
                targets = torch.clone(torch.reshape(target, (-1, 1)).float())
            else:
                preds = torch.cat((preds, pred))
                targets = torch.cat((targets, torch.reshape(target, (-1, 1)).float()))
        # reg_loss = 0
        # for param in model.parameters():
        #    reg_loss += param.norm(1) # torch.sum(torch.abs(param))
    # check results
    print('***** Results on Test Set ******')
    acc = correct / test_loader.dataset.n_samples
    print('Acc = ' + str(acc))
    # save to csv
    results = pd.DataFrame(columns=['True', 'Prediction'])
    results['True'] = targets.numpy().flatten().astype(float)
    results['Prediction'] = preds.numpy().flatten().astype(float)
    # outer_fold_number = data_path.split('/')[-2][-1]
    # X_loaded = np.loadtxt(data_path + 'test_X_' + outer_fold_number + '_full.csv', delimiter=",")
    results.index = test_indices  # X_loaded[:, 0].astype(int)
    path_to_save = \
        data_path[:data_path.rfind('/')] + '/outerFold' + str(outer_fold_number) + '/predictions/'
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
    base_path = '/bit_storage/Workspace/Maura/ML4RG/'

    parser = argparse.ArgumentParser()
    parser.add_argument("-trials", "--n_trials", type=int, default=50,
                        help="number of trials for optuna")
    parser.add_argument("-baseset", "--base_dataset", type=str, default="atwell_LeafRoll_s",
                        help="base dataset to use")
    parser.add_argument("-kfolds", "--kfolds", type=int, default=3,
                        help="number of folds inner loop")
    parser.add_argument("-nepochs", "--nepochs", type=int, default=500,
                        help="number of epochs")
    parser.add_argument("-l1", '--l1_loss', type=bool, default=False)
    parser.add_argument("-pca", '--pca_decomp', type=bool, default=False)

    args = parser.parse_args()
    n_trials = args.n_trials
    baseset = args.base_dataset
    k_folds = args.kfolds
    n_epochs = args.nepochs
    l1_loss_added = args.l1_loss
    pca = args.pca_decomp
    """
    if pca:
        k_folds = 1
        print("PCA can only be combined with k_folds=1")
    """
    seed_all(42)
    optuna.logging.set_verbosity(optuna.logging.DEBUG)

    for outer_fold_number in [0]: #[0, 1, 2, 3 4]
        print('----Outer Fold ' + str(outer_fold_number) + '-----')
        db_name = 'mlp_class' + str(n_trials) + '-' + str(k_folds) + '-' + str(n_epochs) + str(l1_loss_added) \
                  + str(pca) + baseset + str(outer_fold_number)
        storage = optuna.storages.RDBStorage(
            "sqlite:////home/fhaselbeck/" + db_name + ".db", heartbeat_interval=1,
            failed_trial_callback=restart_from_checkpoint
            )
        study_name = "mlp_ml4rg_encoded_trials-" + str(n_trials) + '_kfolds-' + str(k_folds) + '_epochs-' \
                     + str(n_epochs) + '_l1loss-' + str(l1_loss_added) + '_pca-' + str(pca) + '_' + baseset \
                     + str(outer_fold_number)
        study = optuna.create_study(
            storage=storage, study_name=study_name,
            direction="maximize", load_if_exists=True
        )
        # data_path = base_path + 'train_test_split/' + baseset + '/outerFold' + str(outer_fold_number) + '/'
        if 'copy' not in baseset:
            data_path = base_path + 'train_test_split/' + baseset + '/' + baseset + '_strat.h5'
        else:
            data_path = base_path + 'train_test_split/' + baseset.split('-')[1] + '/' + baseset + '_strat.h5'
        study.optimize(lambda trial: objective(trial, device=device, data_path=data_path,
                                               outer_fold_number=outer_fold_number, l1_loss_added=l1_loss_added,
                                               k_folds=k_folds, n_epochs=n_epochs, pca=pca),
                       n_trials=n_trials, show_progress_bar=True)

        pruned_trials = study.get_trials(states=(optuna.trial.TrialState.PRUNED,))
        complete_trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        # delete val files of not best studies
        path_to_save = \
            data_path[:data_path.rfind('/')] + '/outerFold' + str(outer_fold_number) + '/predictions/'
        all_files = glob.glob(path_to_save + '*' + study_name + '.csv')
        all_files = [file for file in all_files if 'Trial' + str(study.best_trial.number) not in file]
        for f in all_files:
            os.remove(f)

        train_final_model(data_path=data_path, best_trial=trial, device=device, n_epochs=n_epochs,
                          outer_fold_number=outer_fold_number, l1_loss_added=l1_loss_added, pca=pca)

