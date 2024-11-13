import os
import csv
import torch
import time
import pickle
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, visual_loss, result_print



class Exp(Exp_Basic):
    def __init__(self, args):
        super(Exp, self).__init__(args)

    def _build_model(self):
        # initiate model
        model = self.model_dict[self.args.model].Model(self.args, self.device).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    
    def _get_data(self, type):
        # load data
        datas_pickle_path = os.path.join(self.args.data_path, "pickles", f"{self.args.data}_{self.args.model_id}_train{self.args.train_ratio}_batch{self.args.batch_size}_{self.args.seq_len}_{self.args.label_len}_{self.args.pred_len}", f"{type}_{self.args.data}_datas.pkl")
        loaders_pickle_path = os.path.join(self.args.data_path, "pickles", f"{self.args.data}_{self.args.model_id}_train{self.args.train_ratio}_batch{self.args.batch_size}_{self.args.seq_len}_{self.args.label_len}_{self.args.pred_len}", f"{type}_{self.args.data}_loaders.pkl")
        if os.path.exists(datas_pickle_path) and os.path.exists(loaders_pickle_path):
            self.args.logger.info("Loading existing datasets and dataloaders...")
            with open(datas_pickle_path, 'rb') as f:
                datas = pickle.load(f)
            with open(loaders_pickle_path, 'rb') as f:
                loaders = pickle.load(f)
        else:
            self.args.logger.info(f"Creating datasets and dataloaders for {datas_pickle_path}...")
            datas, loaders = {}, {}
            datas, loaders = data_provider(self.args, type)
            if not os.path.exists(os.path.join("pickles", f"{self.args.data}_{self.args.model_id}_train{self.args.train_ratio}_batch{self.args.batch_size}_{self.args.seq_len}_{self.args.label_len}_{self.args.pred_len}")):
                os.makedirs(os.path.join("pickles", f"{self.args.data}_{self.args.model_id}_train{self.args.train_ratio}_batch{self.args.batch_size}_{self.args.seq_len}_{self.args.label_len}_{self.args.pred_len}"))
            datas_pickle_folder_path = os.path.dirname(datas_pickle_path)
            if not os.path.exists(datas_pickle_folder_path):
                os.makedirs(datas_pickle_folder_path)
            with open(datas_pickle_path, 'wb') as f:
                pickle.dump(datas, f)
            with open(loaders_pickle_path, 'wb') as f:
                pickle.dump(loaders, f)
            
        total = datas.data_x.shape
        self.args.logger.info(f'total_{type}_len:{total}')
        return datas, loaders


    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
        
        
    def _select_criterion(self):
        return nn.MSELoss()
        # return nn.L1Loss()


    def vali(self, criterion, type='val'):
        _, vali_loaders = self._get_data(type)
        
        preds = []
        trues = []
        trains = []
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_tx, seq_x_date, seq_y_date) in enumerate(vali_loaders):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if not isinstance(batch_tx[0], str):
                    batch_tx = batch_tx.float().to(self.device)
                outputs = self.model(batch_tx, batch_x, batch_x_mark, batch_y, batch_y_mark, seq_x_date, seq_y_date)
                
                outputs = outputs[:, -self.args.pred_len:]
                batch_y = batch_y[:, -self.args.pred_len:].to(self.device)
                train = batch_x.detach().cpu().numpy()
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                
                trains.append(train)
                preds.append(pred)
                trues.append(true)
                
                loss = criterion(pred, true)
                total_loss.append(loss)
        trains = np.concatenate(trains, axis=0)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        self.args.logger.info(f'{type} shape: {preds.shape}, {trues.shape}')
        
        preds = np.nan_to_num(preds)
        trues = np.nan_to_num(trues)
        
        mae, wape = metric(preds, trues, trains, seasonality=self.args.seasonality)
        self.args.logger.info(result_print('{}_results----------mae:{}, wape:{}'.format(type, mae, wape)))
        
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss



    def test(self, setting, test=0):
        test_datas, test_loaders = self._get_data(type='test')
        
        # if test:
        #     self.args.logger.info('loading model')
        #     self.model.load_state_dict(torch.load(os.path.join(self.args.output_path, 'checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        trains = []
        folder_path = os.path.join(self.args.output_path, 'test_results/')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_tx, seq_x_date, seq_y_date) in enumerate(test_loaders):
                #
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if not isinstance(batch_tx[0], str):
                    batch_tx = batch_tx.float().to(self.device)
                #
                outputs = self.model(batch_tx, batch_x, batch_x_mark, batch_y, batch_y_mark, seq_x_date, seq_y_date)
               
                outputs = outputs[:, -self.args.pred_len:]
                batch_y = batch_y[:, -self.args.pred_len:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                
                if test_datas.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_datas.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_datas.inverse_transform(batch_y.squeeze(0)).reshape(shape)
                train = batch_x.detach().cpu().numpy()
                pred = outputs
                true = batch_y
                
                trains.append(train)
                preds.append(pred)
                trues.append(true)
                
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_datas.scale and self.args.inverse:
                        shape = input.shape
                        input = test_datas.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, 0], true[0, :, 0]), axis=0)
                    pd = np.concatenate((input[0, :, 0], pred[0, :, 0]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # concate the first dimension of torch preds elements (B1,Pred_len,D)+(B2,Pred_len,D)->(B1+B2,Pred_len,D)
        trains = np.concatenate(trains, axis=0)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        self.args.logger.info(f'test shape: {preds.shape}, {trues.shape}')
        
        # result save
        folder_path = os.path.join(self.args.output_path, 'results/')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # process nan
        preds = np.nan_to_num(preds)
        trues = np.nan_to_num(trues)
        mae, wape = metric(preds, trues, trains, seasonality=self.args.seasonality)
        self.args.logger.info('test_results----------mae:{}, wape:{}'.format(mae, wape))
        result_summary = os.path.join(self.args.output_path_upper + 'result_comparison.csv')
        file_exists = os.path.isfile(result_summary) and os.path.getsize(result_summary) > 0
        print(f'setting:{setting}')
        with open(result_summary, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            if not file_exists:
                csvwriter.writerow(['exp', 'mae', 'wape'])
            csvwriter.writerow([setting, mae, wape])
   

        np.save(folder_path + 'metrics.npy', np.array([mae, wape]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

 
    
    def train(self, setting):    
        _, train_loaders = self._get_data(type='train')
        # _, vali_loaders = self._get_data(type='val')
        # _, test_loaders = self._get_data(type='test')
        
        if self.args.load_model:
            self.args.logger.info('loading model')
            self.model.load_state_dict(torch.load(self.args.load_model_path, 'checkpoints' + setting, 'checkpoint.pth'))
            
        tensorboardloss_path = os.path.join(self.args.output_path+'/tensorboard_loss/')
        if not os.path.exists(tensorboardloss_path):
            os.makedirs(tensorboardloss_path)

        path = os.path.join(self.args.output_path, 'checkpoints/'+ setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loaders)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, delta=self.args.delta, logger=self.args.logger)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        writer = SummaryWriter(log_dir=tensorboardloss_path+'loss')
        x, train_loss_arr, vali_loss_arr, test_loss_arr = [], [], [], []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            x.append(epoch)
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_tx, seq_x_date, seq_y_date) in enumerate(train_loaders):
                
                iter_count += 1
                model_optim.zero_grad()
                #
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # infer batch_tx type and convert to tensor
                if not isinstance(batch_tx[0], str):
                    batch_tx = batch_tx.float().to(self.device)
                #
                outputs = self.model(batch_tx, batch_x, batch_x_mark, batch_y, batch_y_mark, seq_x_date, seq_y_date)
                outputs = outputs[:, -self.args.pred_len:]
                batch_y = batch_y[:, -self.args.pred_len:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
                if (i + 1) % 100 == 0:
                    self.args.logger.info("iters: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    self.args.logger.info('\t\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                loss.backward()
                model_optim.step()
                torch.cuda.empty_cache()
                
                if i % 10 == 0:  # record every 10 steps
                    writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loaders) + i)

            self.args.logger.info("[Epoch {}] ==== cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(criterion, 'val')
            test_loss = self.vali(criterion, 'test')

            train_loss_arr.append(train_loss)
            vali_loss_arr.append(vali_loss)
            test_loss_arr.append(test_loss)
            self.args.logger.info("[Epoch {0}] Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            #
            writer.add_scalar('Training loss', train_loss, epoch)
            writer.add_scalar('Vali loss', vali_loss, epoch)
            writer.add_scalar('Test loss', test_loss, epoch)
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                self.args.logger.info("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
        writer.close()
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # visualize the loss curve
        visual_loss(x, [train_loss_arr, vali_loss_arr, test_loss_arr], ["train", "validate", "test"], os.path.join(self.args.output_path, setting+'_loss'))
        return self.model
