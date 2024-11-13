import argparse
import os
import torch
import random
import datetime
import numpy as np
from utils.log import Logger
from exp.exp_supervised import Exp

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def config_args():
    parser = argparse.ArgumentParser(description='TFHTS')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='exp', help='model id')
    parser.add_argument('--model', type=str, default='TFHTS', help='model name, options: TFHTS')
    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default=os.getcwd(), help='root path of the data file')
    parser.add_argument('--data_path', type=str, default=os.getcwd()+'/datasets/Wiki-People_en/', help='data dir, options: [PixelLens, Wiki_people, taobao_fashion, News]')
    parser.add_argument('--output_path', type=str, default=os.getcwd()+'/output/', help='output path')
    # parser.add_argument('--root_path', type=str, default='/home/xinz/ar57_scratch/xinz/MMTS', help='root path of the data file')
    # parser.add_argument('--data_path', type=str, default='/home/xinz/ar57_scratch/xinz/MMTS'+'/MMTS_datasets/Wiki_people/', help='data file')
    # parser.add_argument('--output_path', type=str, default='/home/xinz/ar57_scratch/xinz/MMTS'+'/MMTS_output/', help='output path')
    parser.add_argument('--data_ts_filename', type=str, default='train_1_people_en_filtered.csv', help='data filename')
    parser.add_argument('--data_tx_filename', type=str, default='txt_avg_emb.npz', help='data filename')
    parser.add_argument('--freq', type=str, default='d', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--load_model', action='store_true', default=False, help='resume model')
    parser.add_argument('--load_model_path', type=str, default='', help='resume model path')
    parser.add_argument('--channel_independent', action='store_true', default=True, help='channel_independent')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='train ratio')
    # model
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--activation2', type=str, default='gelu', help='activation for model, to be set, now fixed')
    parser.add_argument('--projection_dims', type=int, nargs='+', default=[768, 768], help='hidden layer dimensions of projector (List)')
    parser.add_argument('--output_attention2', action='store_true', default=False, help='output attention in encoder')
    # ts model
    parser.add_argument('--ts_model', type=str, default='PatchTST', help='time series model name, options: [PatchTST, iTransformer]')
    parser.add_argument('--seq_len', type=int, default=7, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=7, help='prediction sequence length')
    parser.add_argument('--seasonality', type=int, default=7, help='seasonality')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--revin', action='store_true', default=False, help='whether to use revin')
    # patchtst
    parser.add_argument('--patch_len', type=int, default=1, help='length of patch')
    parser.add_argument('--patch_stride', type=int, default=1, help='stride of patch')
    parser.add_argument('--n_heads', type=int, default=16, help='num of heads')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout for patchtst model')
    parser.add_argument('--activation1', type=str, default='gelu', help='activation for patchtst')
    parser.add_argument('--output_attention1', action='store_true', default=False, help='output attention in encoder')
    parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')
    # text
    parser.add_argument('--d_txt', type=int, default=4096, help='dimension of text')
    parser.add_argument('--txt_rep', type=str, default='avg', help='text representation, options:[cls, avg, bos, eos]')
    parser.add_argument('--fusion_head', type=int, default=8, help='num of heads in fused cross attention')
    # parser.add_argument('--txtmodel_id', type=str, default='meta-llama/Meta-Llama-3.1-8B', help='text model id')
    # training
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--delta', type=int, default=0.00001, help='early stopping delta')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lr_adj', type=str, default='type2', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    # GPU
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')
    
    return parser.parse_args()
    
    
if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # config
    args = config_args()
    # gpu
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    # current time
    current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    args.current_time = current_time
    setting='{}_{}_data{}_sl{}_ll{}_pl{}_lr{}_lr_adj{}_bs{}_df{}_dm{}_nh{}_fh{}_el{}_revin{}_d{}_{}'.format(
            args.model,
            args.model_id,
            args.data,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.learning_rate,
            args.lr_adj,
            args.batch_size,
            args.d_ff,
            args.d_model,
            args.n_heads,
            args.fusion_head,
            args.e_layers,
            args.revin,
            args.dropout,
            args.current_time)
        
    # mkdir for the current experiment
    args.output_path_upper = args.output_path
    args.output_path = os.path.join(args.output_path, setting)
    os.makedirs(args.output_path)
    
    # create log
    args.logger = Logger(args.output_path, 'log')
    args.logger.info('Args in experiment:')
    args.logger.info(str(args))
    
    # train model
    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            new_setting = 'train_ii{}'.format(ii)
            args.logger.info(f'>>>>>>>start training : {new_setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(new_setting)
            #
            args.logger.info(f'>>>>>>>testing : {new_setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting+'_'+new_setting)
            torch.cuda.empty_cache()
    else:
        args.logger.info(f'>>>>>>>testing : <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp = Exp(args)
        print(setting)
        exp.test(setting, test=1)
        torch.cuda.empty_cache()

                              