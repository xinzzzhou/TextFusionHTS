import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.timefeatures import time_features
from sklearn.preprocessing import StandardScaler
import ast

class BasicDataset(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='ts_data/', data_ts_filename='8M_series_daily_filtered_cut.csv', 
                 data_tx_filename='image_text_info.csv', features='M', 
                 scale=True, freq='d', ratios=[0.7,0.2], channel_independent=True):
        super(BasicDataset, self).__init__()
        if size == None:
            raise Exception("Please indicate the seq_len, label_len, pred_len")
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag
        self.scale = scale
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.data_ts_filename = data_ts_filename
        self.data_tx_filename = data_tx_filename
        
        self.channel_independent = channel_independent
        self.features = features
        self.train_ratio = ratios[0]
        self.test_ratio = ratios[1]
        
        self.__read_data__()
        
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        
        if self.tot_len < 0:
            return None
        
        # self.txt_date = None
    
    def __read_data__(self):
        self.scaler = StandardScaler()
        
        # load time series data
        df_ts_raw = pd.read_csv(os.path.join(self.data_path, self.data_ts_filename))
        print(self.train_ratio, self.test_ratio)
        num_train = int(len(df_ts_raw) * self.train_ratio)
        num_test = int(len(df_ts_raw) * self.test_ratio)
        num_vali = len(df_ts_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_ts_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_ts_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_ts_raw.columns[1:]
        df_data = df_ts_raw[cols_data]
        
        # 
        if self.scale:
            if 'News' not in self.data_path:
                # get log(x+1) to avoid dominate by 0 in PixelRec, dataset density is 0.49
                # get log(x+1) to avoid evaluator in wikipedia
                df_data = np.log1p(df_data)
            # scale
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_ts_raw[['Date']][border1:border2]
        df_txt_date = df_stamp['Date']
        if 'News' in self.data_path:
            data_stamp = np.zeros((len(df_stamp), 1))
        else:
            df_stamp['Date'] = pd.to_datetime(df_stamp.Date)
            data_stamp = time_features(pd.to_datetime(df_stamp['Date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.txt_date = df_txt_date[border1:border2].tolist()    
        self.data_stamp = data_stamp
    
    def __len__(self):
        return max(self.tot_len * self.enc_in, 0)
    
    def __getitem__(self, index):   
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
    
        if self.channel_independent:
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            if self.features == 'M' or self.features == 'MS':
                seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
            elif self.features == 'S':
                seq_y = self.data_y[r_begin:r_end]
        else:
            seq_x = self.data_x[s_begin:s_end, :]
            seq_y = self.data_y[r_begin:r_end, :]
        
        seq_x_date = self.txt_date[s_begin:s_end]
        seq_y_date = self.txt_date[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_x = torch.tensor(seq_x, dtype=torch.float32)
        seq_y = torch.tensor(seq_y, dtype=torch.float32)

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_date, seq_y_date
        
class Dataset_txemb_npz(BasicDataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='ts_data/', data_ts_filename='8M_series_daily_filtered_cut.csv', 
                 data_tx_filename='image_text_info.csv', features='M', 
                 scale=True, freq='d', ratios=[0.7,0.2], channel_independent=True):
        super().__init__(root_path, flag, size, data_path, data_ts_filename, data_tx_filename, features, scale, freq, ratios, channel_independent)

    def __read_data__(self):
        super().__read_data__()
        # load text data (ensure the same order with time series data)
        # load .npz file
        np_tx_raw = np.load(os.path.join(self.data_path, self.data_tx_filename))
        np_tx_raw = np_tx_raw['arr_0']
        # transfer np_tx_raw to tensor
        self.data_tx = []
        for i in range(len(np_tx_raw)):
            tensor = torch.tensor(np_tx_raw[i], dtype=torch.float32)
            self.data_tx.append(tensor)
        
       
    def __getitem__(self, index):
        seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_date, seq_y_date = super().__getitem__(index)
        feat_id = index // self.tot_len
        
        seq_tx = self.data_tx[feat_id]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_tx, seq_x_date, seq_y_date


class Dataset_txemb(BasicDataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='ts_data/', data_ts_filename='.csv', 
                 data_tx_filename='image_text_info.csv', features='M', 
                 scale=True, freq='d',  ratios=[0.7,0.2], channel_independent=True):
        super().__init__(root_path, flag, size, data_path, data_ts_filename, data_tx_filename, features, scale, freq, ratios, channel_independent)

    def __read_data__(self):
        super().__read_data__()
        
        #
        
        # load text data (ensure the same order with time series data)
        df_tx_raw = pd.read_csv(os.path.join(self.data_path, self.data_tx_filename))
        
        df_tx_raw = df_tx_raw[['text_embedding']]
        # transfer df_tx_raw to list
        self.data_tx = []
        for _, row in df_tx_raw.iterrows():
            text_embedding_str = row['text_embedding']
            embedding_list = ast.literal_eval(text_embedding_str)[0]
            tensor = torch.tensor(embedding_list, dtype=torch.float32)
            self.data_tx.append(tensor)

    def __getitem__(self, index):
        seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_date, seq_y_date = super().__getitem__(index)
        feat_id = index // self.tot_len
        
        seq_tx = self.data_tx[feat_id]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_tx, seq_x_date, seq_y_date

