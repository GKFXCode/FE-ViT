from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import re
import logging
import torch
from ..dataset import get_dataset
from ..optimizer import get_optimizer
from ..runner import RUNNER
from .base_runner import BaseRunner
import os
import numpy as np
import copy
from omegaconf import OmegaConf,open_dict
import collections

@RUNNER.register('HREstimateRunner')
class HREstimateRunner(BaseRunner):

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
                pred = model(data[input_type].to(device))
                # print(pred)
                loss = loss_func(pred, data[gt_type].to(device))
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
        
    @torch.no_grad()
    def eval(self, model):
        cfg = copy.copy(self.cfg)
        with open_dict(cfg):
            cfg.dataset.val.return_video_code = True
        valset = get_dataset(cfg, mode='val')
        valloader = DataLoader(valset, batch_size=self.cfg.run.batch_size_val, shuffle=False, num_workers=4)
        gt_type = valset.gt_type
        input_type = self.cfg.dataset.val.input_type
        
        device = self.deal_with_device(model)
        preds = []
        gts = []
        video_code = []
        for data in tqdm(valloader):
            pred = model(data[input_type].to(device))
            pred = torch.mean(pred, axis=1)
            pred = pred.cpu().numpy()
            
            gt =  data[gt_type]
            gt = torch.mean(gt, axis=1)
            gt = gt.cpu().numpy()
      
            preds.append(pred)
            gts.append(gt)
            video_code.append(data['video_code'])
            
        preds = np.hstack(preds)
        gts = np.hstack(gts)
        video_code = np.hstack(video_code)
        np.save('preds.npy', preds)
        np.save('gts.npy', gts)
        np.save('video_code.npy', video_code)
    
    