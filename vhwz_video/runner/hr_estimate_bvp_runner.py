from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import re
import logging
import torch
from ..dataset import get_dataset
from ..optimizer import get_optimizer
from . import RUNNER
from ..metric.hr_metric import get_hr_metrics, plot_analyse_results
from .base_runner import BaseRunner
import os
import numpy as np
import copy
from omegaconf import OmegaConf,open_dict
import collections
import torch.nn as nn
from einops import rearrange


import numpy as np
import scipy
import scipy.io
from scipy.signal import butter
from scipy.sparse import spdiags


def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def _detrend(input_signal, lambda_value):
    """Detrend PPG signal."""
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    detrended_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return detrended_signal

def mag2db(mag):
    """Convert magnitude to db."""
    return 20. * np.log10(mag)

def _calculate_fft_hr(ppg_signal, fs=60, low_pass=0.75, high_pass=2.5):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr

def _calculate_peak_hr(ppg_signal, fs):
    """Calculate heart rate based on PPG using peak detection."""
    ppg_peaks, _ = scipy.signal.find_peaks(ppg_signal)
    hr_peak = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
    return hr_peak

def _calculate_SNR(pred_ppg_signal, hr_label, fs=30, low_pass=0.75, high_pass=2.5):
    """Calculate SNR as the ratio of the area under the curve of the frequency spectrum around the first and second harmonics 
        of the ground truth HR frequency to the area under the curve of the remainder of the frequency spectrum, from 0.75 Hz
        to 2.5 Hz. 

        Args:
            pred_ppg_signal(np.array): predicted PPG signal 
            label_ppg_signal(np.array): ground truth, label PPG signal
            fs(int or float): sampling rate of the video
        Returns:
            SNR(float): Signal-to-Noise Ratio
    """
    # Get the first and second harmonics of the ground truth HR in Hz
    first_harmonic_freq = hr_label / 60
    second_harmonic_freq = 2 * first_harmonic_freq
    deviation = 6 / 60  # 6 beats/min converted to Hz (1 Hz = 60 beats/min)

    # Calculate FFT
    pred_ppg_signal = np.expand_dims(pred_ppg_signal, 0)
    N = _next_power_of_2(pred_ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(pred_ppg_signal, fs=fs, nfft=N, detrend=False)

    # Calculate the indices corresponding to the frequency ranges
    idx_harmonic1 = np.argwhere((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation)))
    idx_harmonic2 = np.argwhere((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation)))
    idx_remainder = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass) \
     & ~((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation))) \
     & ~((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation))))

    # Select the corresponding values from the periodogram
    pxx_ppg = np.squeeze(pxx_ppg)
    pxx_harmonic1 = pxx_ppg[idx_harmonic1]
    pxx_harmonic2 = pxx_ppg[idx_harmonic2]
    pxx_remainder = pxx_ppg[idx_remainder]

    # Calculate the signal power
    signal_power_hm1 = np.sum(pxx_harmonic1)
    signal_power_hm2 = np.sum(pxx_harmonic2)
    signal_power_rem = np.sum(pxx_remainder)

    # Calculate the SNR as the ratio of the areas
    if not signal_power_rem == 0: # catches divide by 0 runtime warning 
        SNR = mag2db((signal_power_hm1 + signal_power_hm2) / signal_power_rem)
    else:
        SNR = 0
    return SNR

def calculate_metric_per_video(predictions, labels, fs=20, diff_flag=True, use_bandpass=True, hr_method='FFT'):
    """Calculate video-level HR and SNR"""
    if diff_flag:  # if the predictions and labels are 1st derivative of PPG signal.
        predictions = _detrend(np.cumsum(predictions), 100)
        labels = _detrend(np.cumsum(labels), 100)
    else:
        predictions = _detrend(predictions, 100)
        labels = _detrend(labels, 100)
    if use_bandpass:
        # bandpass filter between [0.75, 2.5] Hz
        # equals [45, 150] beats per min
        [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
        predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
        labels = scipy.signal.filtfilt(b, a, np.double(labels))
    if hr_method == 'FFT':
        hr_pred = _calculate_fft_hr(predictions, fs=fs)
    elif hr_method == 'Peak':
        hr_pred = _calculate_peak_hr(predictions, fs=fs)
    else:
        raise ValueError('Please use FFT or Peak to calculate your HR.')
    return None, hr_pred, None



@RUNNER.register('HREstimateBVPRunner')
class HREstimateBVPRunner(BaseRunner):

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
            
        loss_func = nn.MSELoss()
        
        start_epoch = 0
        if self.cfg.run.ckpt:
            start_epoch = int(re.search('(?<=epoch_)\d*(?=.pth)', self.cfg.run.ckpt).group()) + 1
        pbar = tqdm(range(start_epoch, self.cfg.run.epochs))
        
        for epoch in pbar:
            for i, data in enumerate(trainloader):
                break
                raw_input = data[input_type] # N, C, T, H, W
                raw_input = torch.transpose(raw_input, 1, 2) # N, T, C, H, W
                diff_input = raw_input[:, 1:] - raw_input[:,:-1] # N, T, C, H, W
                raw_input = raw_input[:, 1:] # 配合帧差分，去掉第一个值
                raw_input = rearrange(raw_input, 'n t c h w -> (n t) c h w')
                diff_input = rearrange(diff_input, 'n t c h w -> (n t) c h w')
                labels = data[gt_type][:, 1:]
                labels = rearrange(labels, 'n t -> (n t) 1')
                pred = model(diff_input.to(device), raw_input.to(device))
                # print(pred)
                loss = loss_func(pred, labels.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                self.writer.add_scalar('Loss/train', loss.item(), epoch*len(trainloader)+i)
                
                pbar.set_description('epoch:%d, iter:%d / %d, loss:%.5f'%(
                    epoch, i, len(trainloader), loss.data))
                
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
            raw_input = data[input_type] # N, C, T, H, W
            N, C, T, H, W = raw_input.shape
            raw_input = torch.transpose(raw_input, 1, 2) # N, T, C, H, W
            diff_input = raw_input[:, 1:] - raw_input[:,:-1] # N, T, C, H, W
            raw_input = raw_input[:, 1:] # 配合帧差分，去掉第一个值
            raw_input = rearrange(raw_input, 'n t c h w -> (n t) c h w')
            diff_input = rearrange(diff_input, 'n t c h w -> (n t) c h w')
            
            pred = model(diff_input.to(device), raw_input.to(device))
    
            pred = pred.view(N, T-1)
            pred = pred.cpu().numpy() # pred bvp
            

            
            for bvp in data['bvps'][:,1:]:
                hr_label, hr_pred, SNR = calculate_metric_per_video(bvp, bvp, fs=20, diff_flag=False, hr_method='FFT')
                preds.append(hr_pred)
            
            gt =  data['hrs']
            gt = torch.mean(gt, axis=1)
            gt = gt.cpu().numpy()
      
            
            gts.append(gt)
            video_code.append(data['video_code'])
            
        preds = np.hstack(preds)
        gts = np.hstack(gts)
        video_code = np.hstack(video_code)
        np.save('preds.npy', preds)
        np.save('gts.npy', gts)
        np.save('video_code.npy', video_code)
        # gts = 5*gts
        # preds = 5*preds
        video_pred = collections.defaultdict(list)
        video_gt = collections.defaultdict(list)
        if True: # eval by video
            for p, g, v in zip(preds, gts, video_code):
                video_pred[v].append(p)
                video_gt[v].append(g)
                
        preds = np.array([np.mean(p) for p in video_pred.values()])       
        gts = np.array([np.mean(g) for g in video_gt.values()])
        
        # 过滤数据集中的异常数据
        idx = np.where((gts>=50) & (gts<=150))
        preds = preds[idx]
        gts = gts[idx]
        res = get_hr_metrics(preds, gts)
        fig = plot_analyse_results(preds, gts)
        fig.savefig('res.png')
        return res
    