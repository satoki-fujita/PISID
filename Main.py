import os
import argparse
import numpy as np
from datetime import datetime
from torch import optim
from torch.nn import functional as F
import torch
from utils import Utils
from model.PISID import PISID


class ModelTrainer(object):
    def __init__(self, params: dict):
        self.params = params
        self.model = self.get_model().to(params['GPU'])
        self.criterion = self.get_loss()
        self.optimizer = self.get_optimizer()
        self.clip = 5
        self.task_level = 1
        self.cl = True

    def get_model(self):
        model = PISID(num_nodes = self.params['N'], in_dim=self.params['in_dim'], in_len=self.params['obs_len'], out_len=self.params['pred_len'])
        return model

    def get_loss(self):
        if self.params['loss'] == 'MSE':
            criterion = Utils.masked_mse
        elif self.params['loss'] == 'MAE':
            criterion = Utils.masked_mae
        else:
            raise NotImplementedError('Invalid loss function.')
        return criterion

    def get_optimizer(self):
        if self.params['optimizer'] == 'Adam':
            optimizer = optim.Adam(params=self.model.parameters(),
                                   lr=self.params['learn_rate'],
                                   weight_decay=self.params['weight_decay'])
        else:
            raise NotImplementedError('Invalid optimizer name.')
        return optimizer

    def train(self, data_loader: dict, modes: list, early_stop_patience=20):
        val_loss = np.inf
        patience_count = early_stop_patience

        print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        print('model training begins:')

        for epoch in range(1, 1 + self.params['num_epochs']):
            starttime = datetime.now()
            running_loss = {mode: 0.0 for mode in modes}
            for mode in modes:
                if mode == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                step = 0

                for x_node, x_state, y_true in data_loader[mode]:
                    with torch.set_grad_enabled(mode=(mode == 'train')):                        
                        y_pred = self.model(x_node, x_state, self.params)

                        if mode == 'train':
                            self.optimizer.zero_grad()
                            if epoch % 2 == 0 and self.task_level <= self.params['pred_len']:
                                self.task_level += 1
                            if self.cl:
                                loss = self.criterion(y_pred[:, :self.task_level, :, :],
                                                      y_true[:, :self.task_level, :, :], 0.0)
                            else:
                                loss = self.criterion(y_pred, y_true, 0.0)
                            loss.backward()
                            if self.clip is not None:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                            self.optimizer.step()
                        else:
                            loss = self.criterion(y_pred, y_true ,0.0)

                    running_loss[mode] += loss * y_true.shape[0]
                    step += y_true.shape[0]

                if mode == 'validate':
                    epoch_val_loss = running_loss[mode] / step
                    if epoch_val_loss <= val_loss:
                        print(f'Epoch {epoch}, validation loss drops from {val_loss:.5} to {epoch_val_loss:.5}. '
                              f'Update model checkpoint..', f'used {(datetime.now() - starttime).seconds}s')
                        val_loss = epoch_val_loss
                        torch.save(self.model.state_dict(), self.params['output_dir'] + '/model_od.pkl')
                        patience_count = early_stop_patience
                    else: # early stopping
                        print(f'Epoch {epoch}, validation loss does not improve from {val_loss:.5}.',
                              f'used {(datetime.now() - starttime).seconds}s')
                        patience_count -= 1
                        if patience_count == 0:
                            print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
                            print(f'    Early stopping at epoch {epoch}. model training ends.')
                            return
        print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        print('model training ends.')
        return

    def test(self, data_loader: dict, modes: list):
        trained_checkpoint = torch.load(self.params['output_dir'] + '/model_od.pkl')
        self.model.load_state_dict(trained_checkpoint)
        self.model.eval()

        for mode in modes:
            print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
            print(f'model testing on {mode} data begins:')
            forecast, ground_truth = [], []
            for x_node, x_state, y_true in data_loader[mode]:
                y_pred = self.model(x_node, x_state, self.params)

                forecast.append(y_pred.cpu().detach())
                ground_truth.append(y_true.cpu().detach())

            forecast = torch.cat(forecast, dim=0)
            ground_truth = torch.cat(ground_truth, dim=0)

            if mode == 'test':
                # evaluate on metrics
                evals_T_all = self.evaluate(forecast, ground_truth)
                self.test_results = {eval_key:[evals_T_all[eval_key]] for eval_key in evals_T_all.keys()}
                for T_i in range(ground_truth.shape[1]):
                    evals_T_i = self.evaluate(forecast[:, T_i, :, :], ground_truth[:, T_i, :, :])
                    for eval_key in self.test_results.keys():
                        self.test_results[eval_key].append(evals_T_i[eval_key])

        print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        print('model testing ends.')
        return

    @staticmethod
    def evaluate(y_pred: np.array, y_true: np.array):
        def MAE(y_pred: np.array, y_true: np.array):
            return F.l1_loss(y_pred, y_true)
        
        def RMSE(y_pred: np.array, y_true: np.array):
            return torch.sqrt(F.mse_loss(y_pred, y_true))

        def MAPE(y_pred: np.array, y_true: np.array):
            y_true_nonzero = y_true.flatten()[y_true.flatten()!=0]
            y_pred_nonzero = y_pred.flatten()[y_true.flatten()!=0]
            return torch.mean(torch.abs(y_pred_nonzero-y_true_nonzero)/y_true_nonzero)

        def RAE(y_pred: np.array, y_true: np.array):
            loss = torch.abs(y_pred-y_true)/(torch.abs(y_true-y_true.mean()).sum())
            return torch.sum(loss)
        
        def PCC(y_pred: np.array, y_true: np.array):
            y_pred_mean = torch.mean(y_pred)
            y_true_mean = torch.mean(y_true)
            top = ((y_pred - y_pred_mean) * (y_true - y_true_mean)).sum()
            bottom = pow(((y_true - y_true_mean) ** 2).sum(), 1 / 2) * pow(((y_pred - y_pred_mean) ** 2).sum(), 1 / 2)
            pcc = top / bottom
            return pcc
        
        def CCC(y_pred: np.array, y_true: np.array):
            y_pred = y_pred.reshape([-1])
            y_true = y_true.reshape([-1])
            cor = np.corrcoef(y_pred, y_true)[0, 1]
            y_pred_mean = torch.mean(y_pred)
            y_true_mean = torch.mean(y_true)
            y_pred_std = torch.std(y_pred)
            y_true_std = torch.std(y_true)
            y_pred_var = torch.var(y_pred)
            y_true_var = torch.var(y_true)
            top = 2 * cor * y_pred_std * y_true_std
            bottom = y_pred_var + y_true_var + (y_pred_mean - y_true_mean) ** 2
            ccc = top / bottom
            return ccc
        return {'MAE':MAE(y_pred, y_true), 'RMSE':RMSE(y_pred, y_true), 'MAPE':MAPE(y_pred, y_true), 'RAE':RAE(y_pred, y_true), 'PCC':PCC(y_pred, y_true), 'CCC':CCC(y_pred, y_true)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Prediction')
    parser.add_argument('-GPU', '--GPU', type=str, help='Specify GPU usage', default='cuda:0')
    parser.add_argument('-in', '--input_dir', type=str, default='./data')
    parser.add_argument('-out', '--output_dir', type=str, default='./output')
    parser.add_argument('-data', '--used_data', type=str, help='Specify data', choices=['JP','US'], default='JP')
    parser.add_argument('-obs', '--obs_len', type=int, help='Length of observation sequence', default=28)
    parser.add_argument('-pred', '--pred_len', type=int, help='Length of prediction sequence', default=28)
    parser.add_argument('-dim', '--in_dim', type=int, help='Dimension of input features', default=1)
    parser.add_argument('-split', '--split_ratio', type=float, nargs='+',
                        help='Relative data split ratio in train : validate : test'
                             ' Example: -split 6 1 1', default=[6, 1, 3])
    parser.add_argument('-batch', '--batch_size', type=int, default=32)
    parser.add_argument('-epoch', '--num_epochs', type=int, default=300)
    parser.add_argument('-loss', '--loss', type=str, help='Specify loss function',
                        choices=['MSE', 'MAE'], default='MAE')
    parser.add_argument('-optim', '--optimizer', type=str, help='Specify optimizer', default='Adam')
    parser.add_argument('-lr', '--learn_rate', type=float, default=1e-3)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-8)
    parser.add_argument('-rep', '--reproduce', type=bool, default=True)
    params = parser.parse_args().__dict__

    os.makedirs(params['output_dir'], exist_ok=True)

    # load data
    data_input = Utils.DataInput(data_dir=params['input_dir'], use_data=params['used_data'])
    data_dic = data_input.load_data()    
    data_loader_dic=dict()
    for key in data_dic.keys():
        params['N'] = data_dic[key]['node'].shape[1]     
        # get data loader
        data_generator = Utils.DataGenerator(obs_len=params['obs_len'],
                                             pred_len=params['pred_len'],
                                             data_split_ratio=params['split_ratio'])
        data_loader, _ = data_generator.get_data_loader(data=data_dic[key], params=params)
        data_loader_dic[key] = data_loader
 
    evaluates_dic = {}
    for key in data_dic.keys():
        evaluates = {'MAE':[],'RMSE':[],'MAPE':[],'RAE':[],'PCC':[],'CCC':[]}
        for k in range(5):
            Utils.seed_torch(k)
            trainer = ModelTrainer(params=params)
            trainer.train(data_loader=data_loader_dic[key], modes=['train', 'validate'])
            trainer.test(data_loader=data_loader_dic[key], modes=['test'])            
            for eval in evaluates.keys():
                evaluates[eval].append(trainer.test_results[eval])

        for eval in evaluates.keys():
            evaluates[eval] = np.array(evaluates[eval])
        evaluates_dic[key] = evaluates
    for key in data_dic.keys():
        print(params['used_data']+': '+key)
        for eval in evaluates_dic[key].keys():
            print(eval + f': {np.mean(evaluates_dic[key][eval],0)[-1]} ({np.std(evaluates_dic[key][eval],0)[-1]})')