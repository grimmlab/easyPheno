import numpy as np
import torch
from torch.utils.data import Dataset
import h5py


class SNPDataset(Dataset):
    def __init__(self, train_test, k, data_path, change_encoding=True):
        outer_fold_number = data_path.split('/')[-2][-1]
        X_loaded = np.loadtxt(data_path + train_test + '_X_' + outer_fold_number + '_' + str(k) + '.csv', delimiter=",")
        if change_encoding:
            X_loaded[:, 1:] = np.where(X_loaded[:, 1:] < 1, 0, 1)
        y_loaded = np.loadtxt(data_path + train_test + '_y_' + outer_fold_number + '_' + str(k) + '.csv', delimiter=",")
        self.X = torch.from_numpy(X_loaded[:, 1:])
        self.y = torch.from_numpy(y_loaded[:, 1])
        self.n_samples = self.y.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SNPDatasetH5(Dataset):
    def __init__(self, h5_path, outerfold_idx, innerfold_idx, train_val_str, b_onehot=False, change_encoding=True):
        with h5py.File(h5_path, "r") as f:             
            sid = f[f'outerfold_{outerfold_idx}/innerfold_{innerfold_idx}/{train_val_str}/sid'][:]
            X = f[f'outerfold_{outerfold_idx}/innerfold_{innerfold_idx}/{train_val_str}/X'][:]
            X_onehot = f[f'outerfold_{outerfold_idx}/innerfold_{innerfold_idx}/{train_val_str}/X_onehot'][:]
            y = f[f'outerfold_{outerfold_idx}/innerfold_{innerfold_idx}/{train_val_str}/y'][:]
    
        self.sid = torch.from_numpy(sid)
        self.y = torch.from_numpy(y)
        self.n_samples = len(sid)
        
        if b_onehot:
            self.X = torch.from_numpy(X_onehot)
        else:
            if change_encoding:
                X = np.where(X < 1, 0, 1)
            self.X = torch.from_numpy(X)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
