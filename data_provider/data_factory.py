from data_provider.data_loader import Dataset_txemb_npz
from torch.utils.data import DataLoader


data_dict = {
    'custom': Dataset_txemb_npz,
}


def data_provider(args, flag, logger=None, drop_last_test=False):
    Data = data_dict[args.data]

    if flag == 'test':
        shuffle_flag = False
        drop_last = drop_last_test
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'val':
        shuffle_flag = True
        drop_last = drop_last_test
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'train':
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        raise Exception("unknown flag for dataset")

    args.test_ratio = 1 - args.train_ratio - 0.1
    print(args.test_ratio)
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        data_ts_filename=args.data_ts_filename,
        data_tx_filename=args.data_tx_filename,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        freq=freq,
        ratios=[args.train_ratio, args.test_ratio],
        channel_independent=args.channel_independent)
    print(flag, len(data_set))
    try:
        data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                drop_last=drop_last)
    except ValueError as e:
        print(f"Erro message: {e}")
        return None, None
    return data_set, data_loader
