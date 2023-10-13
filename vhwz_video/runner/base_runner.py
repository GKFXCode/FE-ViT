from ..model import get_model
import torch

class BaseRunner():
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        
    def deal_with_device(self, model):
        device = torch.device(self.cfg.run.device)
        model.to(device)
        return device
    
    def get_model(self):
        model = get_model(self.cfg)
        if self.cfg.run.ckpt:
            print('loading ckpt')
            model.load_state_dict(torch.load(self.cfg.run.ckpt, map_location='cpu'))
        return model