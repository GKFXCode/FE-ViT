import torch

def get_optimizer(cfg, model, **args):
    if getattr(cfg.optimizer, 'optimizer_name', "") == 'SGD':
        print('using SGD')
        return torch.optim.SGD(model.parameters(), lr=cfg.optimizer.lr)
    return torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)