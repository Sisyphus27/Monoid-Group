#!/user/bin/env python
# coding=utf-8
"""
@project : Monoid-Group
@author  : zy
@file   : Recommendations_TT.py
@ide    : PyCharm
@time   : 2022/9/6 11:27
"""
import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from Recommendations.Recommendations_Data_Load import \
    MovieLens1MDataset,MovieLens20MDataset
from DeepFM.DeepFM_Model import DeepFactorizationMachineModel
from Factorization_Machines.Factorization_Machines_model import FactorizationMachineModel
from Field_aware_Factorization_Machines.Field_aware_Factorization_Machines_Model import \
    FieldAwareFactorizationMachineModel
from Attention_Factorization_Machines.Attention_Factorization_Machines_Model import \
    AttentionalFactorizationMachineModel
from Deep_Cross_Network.Deep_Cross_Network_Model import DeepCrossNetworkModel
from Neural_Factorization_Machines.Neural_Factorization_Machines_Model import NeuralFactorizationMachineModel


def get_dataset(name, path):
    try:
        if name == 'movielens1M':
            return MovieLens1MDataset(path)
        elif name == 'movielens20M':
            return MovieLens20MDataset(path)
    except ValueError:
        print("No such {} dataset".format(name))


def get_model(name, dataset: MovieLens1MDataset):
    field_dims = dataset.field_dims
    try:
        if name == 'fm':
            return FactorizationMachineModel(field_dims, embed_dim=16)
        elif name == 'ffm':
            return FieldAwareFactorizationMachineModel(field_dims, embed_dim=4)
        elif name == 'dcn':
            return DeepCrossNetworkModel(field_dims, embed_dim=16, num_layers=3, mlp_dims=(16, 16), dropout=0.2)
        elif name == 'dfm':
            return DeepFactorizationMachineModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
        elif name == 'nfm':
            return NeuralFactorizationMachineModel(field_dims, embed_dim=64, mlp_dims=(64,), dropouts=(0.2, 0.2))
        elif name == 'afm':
            return AttentionalFactorizationMachineModel(field_dims, embed_dim=16, attn_size=16, dropouts=(0.2, 0.2))

    except ValueError:
        print("unknown model name {}".format(name))


class EarlyStopper(object):
    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader, criterion,
          device,
          log_interval=100):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def test(model: nn.Module, data_loader: torch.utils.data.DataLoader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device).long(), target.to(device).long()
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)


def main(dataset_name,
         dataset_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         save_dir):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataset = get_dataset(dataset_name, dataset_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length)
    )
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)

    model = get_model(model_name, dataset).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(num_trials=2, save_path=f'{save_dir}/{model_name}.pt')

    for epoch_i in range(epoch):
        train(model, optimizer, train_data_loader, criterion, device)
        auc = test(model, valid_data_loader, device)
        print('epoch:', epoch_i, 'validation: auc:', auc)
        if not early_stopper.is_continuable(model, auc):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break
    auc = test(model, test_data_loader, device)
    print(f'test auc: {auc}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='movielens20M')
    parser.add_argument('--dataset_path', default='./ml-20m/ratings.csv',
                        help='criteo/train.txt, avazu/train, or ml-1m/ratings.dat')
    parser.add_argument('--model_name', default='afm')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--save_dir', default='./')
    args = parser.parse_args(args=[])
    print(args.model_name)
    main(args.dataset_name,
         args.dataset_path,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.save_dir)
