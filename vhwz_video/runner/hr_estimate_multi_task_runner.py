from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import re
import logging
import torch
from ..dataset import get_dataset
from ..optimizer import get_optimizer
from ..runner import RUNNER
from ..metric.hr_metric import get_hr_metrics
from .base_runner import BaseRunner
import os
import numpy as np
from .hr_estimate_runner import HREstimateRunner

@RUNNER.register('HREstimateMultiTaskRunner')
class HREstimateMultiTaskRunner(HREstimateRunner):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        return
    
    def train(self):
        self.writer = SummaryWriter()
        trainset = get_dataset(self.cfg, mode='train') 
        trainloader = DataLoader(trainset, batch_size=self.cfg.run.batch_size_train, shuffle=True, num_workers=4)
        gt_type = trainset.gt_type
        input_type = self.cfg.dataset.train.input_type
        
        model = self.get_model()
        device = self.deal_with_device(model)
        model.train()
        
        if hasattr(model, 'optimizer'):
            optimizer = model.optimizer()
        else:
            optimizer = get_optimizer(self.cfg, model)
            
        loss_func = model.loss_function
        
        start_epoch = 0
        if self.cfg.run.ckpt:
            start_epoch = int(re.search('(?<=epoch_)\d*(?=.pth)', self.cfg.run.ckpt).group()) + 1
        pbar = tqdm(range(start_epoch, self.cfg.run.epochs))
        
        for epoch in pbar:
            for i, data in enumerate(trainloader):
                f = model.forward_feature(data[input_type].to(device))
                pred = model.forward_pred(f)
                loss = loss_func(pred, data[gt_type].to(device), f, data['bvps'].to(device), self.writer, epoch*len(trainloader)+i)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                self.writer.add_scalar('Loss/train', loss.item(), epoch*len(trainloader)+i)
                
                pbar.set_description('epoch:%d, iter:%d, loss:%.5f'%(
                    epoch, i, loss.data))
                
            if self.cfg.run.save_ckpt_interval > 0 and (epoch+1)%self.cfg.run.save_ckpt_interval == 0:
                torch.save(model.state_dict(), os.path.join(self.cfg.run.work_dir, 'epoch_%d.pth'%epoch))
                
            if self.cfg.run.eval_interval > 0 and (epoch+1)%self.cfg.run.eval_interval == 0:
                res = self.eval(model)
                logging.info('epoch:%d, %s'%(epoch, res))
                model.train()
        