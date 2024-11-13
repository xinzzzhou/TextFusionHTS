import os
import torch
from models import TFHTS

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TFHTS': TFHTS,}
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        total_params = count_parameters(self.model)
        self.args.logger.info(f"Total trainable parameters: {total_params}")
   
    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            self.args.logger.info('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            self.args.logger.info('Use CPU')
        return device
    

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
