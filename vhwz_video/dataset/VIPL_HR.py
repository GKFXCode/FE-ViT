from torch.utils.data import Dataset
import torchvision
import numpy as np
import torch
from .build import DATASET
import glob
import os
from PIL import Image
import pandas as pd
from scipy.io import loadmat
from einops import rearrange

@DATASET.register('VIPLVideoFrameDataSet')
class VIPLVideoFrameDataSet(Dataset):
    '''
    nb_sample: how many frames should we sammple from a video
    stride: sample one frame for every stride frames 每隔strid帧取一帧， stride=1时连续取。
    '''
    def __init__(self, cfg):
        
        if cfg.dataset.mode == 'train':
            divide_file = cfg.dataset.train.divide_file_train
            args = cfg.dataset.train
        else:
            divide_file = cfg.dataset.val.divide_file_val
            args = cfg.dataset.val
            
        self.gt_type = args.gt_type # hrs or bvps
        self.video_names = self.read_divide_file(divide_file)
        self.videos = {}
        self.root_path = args.root_path
        self.nb_sample = int(args.nb_sample)
        self.stride = int(args.stride_sample)
        video_paths = [(name, os.path.join(self.root_path, name)) for name in self.video_names]
        for name, p in video_paths:
            self.videos[name] = {}
            self.videos[name]['frame_paths'] = sorted(glob.glob(p+'/*.jpg'))
            self.videos[name]['gts'] = p + '/' + name + '.pth'
            
        self.img_transform = torchvision.transforms.ToTensor()
        self.img_size = (int(args.img_size), int(args.img_size))
        self.args = args
        
    def read_divide_file(self, path):
        video_names = []
        with open(path, 'r') as f:
            lines = f.readlines()
        video_names = [L.strip('\n') for L in lines]
        return video_names

    def load_img(self, path):
        img = Image.open(path)
        img = img.resize((self.img_size))
        if self.img_transform:
            img = self.img_transform(img)
        return img

    def load_frames(self, frames_paths, start, nb_sample, stride):
        frames = frames_paths[start:start +(nb_sample - 1)*stride + 2:stride]
        frames = [self.load_img(f) for f in frames]
        frames = torch.stack(frames) # t, c, h, w
        frames = torch.transpose(frames, 0, 1) # c, t, h, w
        return frames

    def load_hrs(self, gt_path, start, nb_sample, stride):
        gt = torch.load(gt_path)
        gt = gt['hrs'][start:start + (nb_sample - 1)*stride + 2:stride]
        return torch.tensor(gt, dtype=torch.float32)/10
    
    def load_bvps(self, path, video_length, start, nb_sample, stride):
        wave = pd.read_csv(path)
        wave = wave['Wave'].values
        wave = wave[0::2][:video_length]
        wave = wave[start:start +(nb_sample - 1)*stride + 2:stride]
        return torch.tensor(wave)/100
            
    def __getitem__(self, index):
        '''
        random choice a start point
        return:
            frames: torch.tensor(b, c, t, h, w)
            hrs: torch.tensor(b, t)
        '''
        name = self.video_names[index]
        frame_paths = self.videos[name]['frame_paths']
        high = len(frame_paths) - (self.nb_sample - 1)*self.stride
        assert high > 0, 'nothing to choice'
        start = np.random.randint(0, high-1)
        frames = self.load_frames(frame_paths, start, self.nb_sample, self.stride)
        # if self.transform is not None:
        #     frames = self.transform(frames)
        data = {'frames': frames}
        
        if self.gt_type == 'bvps':
            p, v, s = name.split('-')    
            path = os.path.join('/media/ubuntu/DATA2/vhwz/dataset/VIPL-HR/', 'data', p, v, 'source'+s[1], 'wave.csv')
            bvps = self.load_bvps(path, len(frame_paths), start, self.nb_sample, self.stride)
            data['bvps'] = bvps
        else:
            hrs = self.load_hrs(self.videos[name]['gts'], start, self.nb_sample, self.stride)
            data['hrs'] = hrs
        return data
    
    def __len__(self):
        return len(self.videos)
   
   
@DATASET.register('VIPLVideoFrameAndDiffDataSet')
class VIPLVideoFrameAndDiffDataSet(Dataset):
    '''
    nb_sample: how many frames should we sammple from a video
    '''
    def __init__(self, cfg):
        
        self.eval_mode = False
        if cfg.dataset.mode == 'train':
            divide_file = cfg.dataset.train.divide_file_train
            args = cfg.dataset.train
        else:
            divide_file = cfg.dataset.val.divide_file_val
            args = cfg.dataset.val
            self.eval_mode = True
            
        self.gt_type = args.gt_type # hrs or bvps
        self.video_names = self.read_divide_file(divide_file)
        self.videos = {}
        self.root_path = args.root_path
        self.nb_sample = int(args.nb_sample)
        video_paths = [(name, os.path.join(self.root_path, name)) for name in self.video_names]
        for name, p in video_paths:
            self.videos[name] = {}
            self.videos[name]['frame_paths'] = sorted(glob.glob(p+'/*.jpg'))
            self.videos[name]['gts'] = p + '/' + name + '.pth'
            
        self.img_transform = torchvision.transforms.ToTensor()
        self.img_size = (int(args.img_size), int(args.img_size))
        self.args = args
        
    def read_divide_file(self, path):
        video_names = []
        with open(path, 'r') as f:
            lines = f.readlines()
        video_names = [L.strip('\n') for L in lines]
        return video_names

    def load_img(self, path):
        img = Image.open(path)
        img = img.resize((self.img_size))
        if self.img_transform:
            img = self.img_transform(img)
        return img

    def load_frames(self, frames_paths, start, nb_sample):
        frames = frames_paths[start:start + nb_sample + 1]
        frames = [self.load_img(f) for f in frames]
        frames = torch.stack(frames) # t, c, h, w
        frames = torch.transpose(frames, 0, 1) # c, t, h, w
        return frames

    def load_hrs(self, gt_path, start, nb_sample):
        gt = torch.load(gt_path)
        gt = gt['hrs'][start:start + nb_sample + 1]
        return torch.tensor(gt, dtype=torch.float32)
    
    def load_bvps(self, path, video_length, start, nb_sample):
        wave = pd.read_csv(path)
        wave = wave['Wave'].values
        wave = wave[0::2][:video_length]
        wave = wave[start:start + nb_sample + 1]
        return torch.tensor(wave)/100
            
    def __getitem__(self, index):
        '''
        random choice a start point
        return:
            frames: torch.tensor(b, c, t, h, w)
            hrs: torch.tensor(b, t)
        '''
        name = self.video_names[index]
        frame_paths = self.videos[name]['frame_paths']
        high = len(frame_paths) - (self.nb_sample + 1)
        assert high > 0, 'nothing to choice'
        start = np.random.randint(0, int(high*0.5))
        frames = self.load_frames(frame_paths, start, self.nb_sample)
        data = {'frames': frames, 'video_code':name}
        
        if self.gt_type == 'bvps':
            p, v, s = name.split('-')    
            path = os.path.join('/media/ubuntu/DATA2/vhwz/dataset/VIPL-HR/', 'data', p, v, 'source'+s[1], 'wave.csv')
            bvps = self.load_bvps(path, len(frame_paths), start, self.nb_sample)
            data['bvps'] = bvps
            if frames.shape[1] != len(bvps):
                print(frames.shape, " ", bvps.shape)
                print('len frame_paths : ', len(frame_paths))
                wave = pd.read_csv(path)
                wave = wave['Wave'].values
                wave = wave[0::2]
                print(len(wave), ' start : ', start, ' high : ', high)
            assert frames.shape[1] == len(bvps), 'error loading frame or bvp, length not match'
        if self.eval_mode:
            hrs = self.load_hrs(self.videos[name]['gts'], start, self.nb_sample)
            data['hrs'] = hrs
        return data
    
    def __len__(self):
        return len(self.videos)
@DATASET.register('VIPLMSTmapDataSet')
class VIPLMSTmapDataSet(VIPLVideoFrameDataSet):
    '''
    input: list of stmap paths, stmaps of a video are save in a .pth with other info as a dict
    output: dict of stmpas:torch.tensor(n, c, h, w) and mean_hr:torch.tensor()
    '''
    def __init__(self, cfg):
        if cfg.dataset.mode == 'train':
            divide_file = cfg.dataset.train.divide_file_train
            args = cfg.dataset.train
        else:
            divide_file = cfg.dataset.val.divide_file_val
            args = cfg.dataset.val
            
        self.video_names = self.read_divide_file(divide_file)
        self.root_path = args.root_path
        self.gt_type = args.gt_type # bpm
        self.img_size = (300, 63)
        self.img_transform = torchvision.transforms.ToTensor()
        
        self.stmap_paths = []
        for name in self.video_names:
            self.stmap_paths += glob.glob(os.path.join(self.root_path, name, '*'))
        
        self.toseq = getattr(args, 'toseq', False)
        self.return_video_code = getattr(args, 'return_video_code', False)
        self.filter_abnormal = getattr(args, 'filter_abnormal', False)
        self.load_bvps = getattr(args, 'load_bvps', False)

    
    def load_mat(self, path, filename):
        data = loadmat(os.path.join(path, filename+'.mat'))[filename][0][0]
        return torch.tensor(data, dtype=torch.float32).view(-1)
    
    def __getitem__(self,index):
        stmap_path = self.stmap_paths[index]
        stmap = self.load_img(os.path.join(stmap_path, 'img_yuv.png')) # c, h, w
        if self.toseq:
            stmap = rearrange(stmap, 'c h w -> w (h c)')
            
        gt = self.load_mat(stmap_path, self.gt_type)
        # 过滤数据集中的异常值
        if self.filter_abnormal: 
            gt = torch.clip(gt, 8, 30)
        res = {'stmap': stmap, self.gt_type: gt}     
        if self.return_video_code:
            video_code = stmap_path.split('/')[-2]
            res['video_code'] = video_code
            
        if self.load_bvps:
            bvps = loadmat(os.path.join(stmap_path, 'bvp'+'.mat'))['bvp'][0]
            res['bvps']= torch.tensor(bvps, dtype=torch.float32).view(-1)/10
        return res
            
    def __len__(self):
        return len(self.stmap_paths)
    

if __name__ == '__main__':
    class Cfg:
        pass
    cfg = Cfg()
    cfg.dataset = Cfg()
    cfg.dataset.root_path = '/media/ubuntu/DATA2/vhwz/dataset/VIPL-HR/frames_cropped'
    cfg.dataset.divide_file_train = '/media/ubuntu/DATA2/vhwz/dataset/VIPL-HR/divide/train_[v1-v9]_[s1].txt'
    cfg.dataset.mode = 'train'
    cfg.dataset.nb_sample = 10
    cfg.dataset.stride_sample = 5
    dataset = VIPLVideoFrameDataSet(cfg)
    dataset[0]
    
    