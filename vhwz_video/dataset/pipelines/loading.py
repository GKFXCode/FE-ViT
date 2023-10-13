import imp
from turtle import pd
from mmcv import video
from torch._C import import_ir_module
from urllib3 import Retry
from ..builder import PIPELINES
import torchvision
import glob
import os
import cv2
import torch
import numpy as np  
import scipy
from scipy.io import loadmat
import pdb

@PIPELINES.register_module()
class LoadFramesFromVideo():
    def __init__(self, color_space='rgb', fps=None, resize=None, to_tensor=False):
        self.color_space = color_space
        self.fps = fps
        self.resize = resize
        self.to_tensor = to_tensor
        self._to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, video_path):
        pass



@PIPELINES.register_module()
class LoadFramesFromImage():
    def __init__(self, color_space='rgb', resize=None, to_tensor=False, max_seq_len=None):
        self.color_space = color_space
        self.resize = resize
        self.to_tensor = to_tensor
        self._to_tensor = torchvision.transforms.ToTensor()
        self.max_seq_len = max_seq_len

    def __call__(self, video_path):
        video_code = video_path.split('/')[-1]
        data_path = os.path.join(video_path, video_code+'.pth')
        image_path = sorted(glob.glob(video_path+'/*.jpg'))
        data = torch.load(data_path)
        mean_hr = data['mean_hr']
        hrs = data['hrs']

        faces = []
        for p in image_path:
            frame = cv2.imread(p)

            if self.resize :
                frame = cv2.resize(frame, self.resize)
            if self.color_space == 'rgb':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif self.color_space == 'yuv':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

            faces.append(frame)

        if self.max_seq_len:
            faces = faces[0:self.max_seq_len]
            hrs = hrs[0:self.max_seq_len]

        if self.to_tensor:
            faces = torch.stack(list(map(self._to_tensor, faces))) #tensor(b,c,h,w)
            hrs = torch.tensor(hrs, dtype=torch.float32)
            mean_hr = torch.tensor(mean_hr, dtype=torch.float32).view(-1)
    
        return {'faces':faces, 'hrs':hrs, 'mean_hr':mean_hr, 'video_code':video_code}

@PIPELINES.register_module()
class LoadFramesFromImageAndPack():
    '''
    output: 
    '''
    def __init__(self, color_space='rgb'):
        self.color_space = color_space

        self._to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, video_path):
        video_code = video_path.split('/')[-1]
        data_path = os.path.join(video_path, video_code+'.pth')
        image_path = sorted(glob.glob(video_path+'/*.jpg'))
        data = torch.load(data_path)
        mean_hr = data['mean_hr']
        hrs = data['hrs']

        faces = []
        for p in image_path:
            frame = cv2.imread(p)

            if self.color_space == 'rgb':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif self.color_space == 'yuv':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

            frame = self._to_tensor(frame)
            frame = torch.nn.functional.adaptive_avg_pool2d(frame, (14,14))

            faces.append(frame)
            
        faces = torch.stack(faces)
        faces = faces[0:16*16]
        hrs = hrs[0:16*16]
        faces = torchvision.utils.make_grid(faces, nrow=16, padding=0, 
                normalize=False, range=None, scale_each=False, pad_value=0)

        if faces.shape[1] != 224 or faces.shape[2] != 224:
            temp = torch.zeros(3,224,224)
            c, h, w = faces.shape
            temp[:, 0:h, 0:w] = faces
            faces = temp

        hrs = torch.tensor(hrs, dtype=torch.float32).mean().view(-1)
        mean_hr = torch.tensor(mean_hr, dtype=torch.float32).view(-1)

        # faces: tensor(c, h, w)
        return {'faces':faces, 'hrs':hrs, 'mean_hr':mean_hr, 'video_code':video_code} 

@PIPELINES.register_module()
class LoadFramesAndMaskFromImage():
    def __init__(self, mask_root, color_space='rgb', fps=None, resize=None, to_tensor=False, max_seq_len=None):
        self.color_space = color_space
        self.fps = fps
        self.resize = resize
        self.to_tensor = to_tensor
        self._to_tensor = torchvision.transforms.ToTensor()
        self.mask_root = mask_root
        self.max_seq_len = max_seq_len

    def __call__(self, video_path):
        video_code = video_path.split('/')[-1]
        data_path = os.path.join(video_path, video_code+'.pth')
        # video_path = sorted(glob.glob(video_path+'/*.jpg'))
        video_frame_paths = sorted(glob.glob(os.path.join(video_path, '*.jpg')))

        mask_paths = sorted(glob.glob(os.path.join(self.mask_root, video_code, '*.png')))

        
        data = torch.load(data_path)
        mean_hr = data['mean_hr']
        hrs = data['hrs']

        faces = []
        for p, q in zip(video_frame_paths, mask_paths):
            frame = cv2.imread(p)
            mask = cv2.imread(q)
            
            if self.resize :
                frame = cv2.resize(frame, self.resize)
            if self.color_space == 'rgb':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif self.color_space == 'yuv':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

            frame = cv2.bitwise_and(frame, frame, mask=mask[...,0])

            faces.append(frame)
        
        if self.max_seq_len:
            faces = faces[0:self.max_seq_len]
            hrs = hrs[0:self.max_seq_len]

        if self.to_tensor:
            faces = self._to_tensor(np.array(faces)) #tensor(b,c,h,w)
            hrs = torch.tensor(hrs, dtype=torch.float32)
            mean_hr = torch.tensor(mean_hr, dtype=torch.float32)
    
        return {'faces':faces, 'hrs':hrs, 'mean_hr':mean_hr, 'video_code':video_code}
        
@PIPELINES.register_module()
class LoadMSTmap():
    def __init__(self, color_space='rgb', gt_type='hr', mode='by_frame'):
        self.color_space = color_space
        self.gt_type = gt_type
        self._to_tensor = torchvision.transforms.ToTensor()
        self.mode = mode

    def __call__(self, path):
        # print(path)
        data = dict()

        if self.mode == 'by_frame':
            data['stmaps'] = cv2.cvtColor(cv2.imread(os.path.join(path,'img_'+self.color_space+'.png')), cv2.COLOR_BGR2RGB)
            if self.gt_type == 'bpm':
                data['stmap_hrs'] = loadmat(os.path.join(path, 'bpm.mat'))['bpm'][0][0]
            elif self.gt_type == 'hr':
                data['stmap_hrs'] = loadmat(os.path.join(path, 'gt.mat'))['gt_temp'][0][0]

            data['stmaps'] = self._to_tensor(data['stmaps'])
            data['stmap_hrs'] = torch.tensor(data['stmap_hrs'], dtype=torch.float32).view(-1)
            data['mean_hr'] = data['stmap_hrs'].view(-1)
            data['video_code'] = path.split('/')[-2]
            data['fps'] = torch.tensor(loadmat(os.path.join(path, 'fps.mat'))['fps'][0][0], dtype=torch.float32)
            data['fps'] = loadmat(os.path.join(path, 'gt.mat'))['gt_temp'][0][0]/loadmat(os.path.join(path, 'bpm.mat'))['bpm'][0][0]
            # print(data['fps'])
        elif self.mode == 'by_video':
            paths = glob.glob(path+'/*')
            stmaps = []
            hrs = []
            
            for p in paths:
                stmaps.append(self._to_tensor(
                    cv2.cvtColor(cv2.imread(os.path.join(p,'img_'+self.color_space+'.png')), cv2.COLOR_BGR2RGB))
                    )
                if self.gt_type == 'bpm':
                    hrs.append(torch.tensor(
                        loadmat(os.path.join(p, 'bpm.mat'))['bpm'][0][0], dtype=torch.float32).view(-1))
                elif self.gt_type == 'hr':
                    hrs.append(torch.tensor(
                        loadmat(os.path.join(p, 'gt.mat'))['gt_temp'][0][0], dtype=torch.float32).view(-1))

            data['stmaps'] = torch.stack(stmaps)
            data['stmap_hrs'] = torch.stack(hrs).view(-1, 1)
            data['mean_hr'] = torch.mean(data['stmap_hrs']).view(-1)
            data['video_code'] = path.split('/')[-1]
            data['fps'] = loadmat(os.path.join(p, 'gt.mat'))['gt_temp'][0][0]/loadmat(os.path.join(p, 'bpm.mat'))['bpm'][0][0]
        
        elif self.mode == 'smooth':
            paths = sorted(glob.glob(path+'/*'), key=lambda x:int(x.split('/')[-1]))
            while len(paths) < 7:
                paths.append(paths[-1])
            start = torch.randint(low=0, high=len(paths)-6, size=(1,))
            paths = paths[start:start+6]

            stmaps = []
            hrs = []
            for p in paths:
                stmaps.append(self._to_tensor(
                    cv2.cvtColor(cv2.imread(os.path.join(p,'img_'+self.color_space+'.png')), cv2.COLOR_BGR2RGB))
                    )
                if self.gt_type == 'bpm':
                    hrs.append(torch.tensor(
                        loadmat(os.path.join(p, 'bpm.mat'))['bpm'][0][0], dtype=torch.float32).view(-1))
                elif self.gt_type == 'hr':
                    hrs.append(torch.tensor(
                        loadmat(os.path.join(p, 'gt.mat'))['gt_temp'][0][0], dtype=torch.float32).view(-1))

            data['stmaps'] = torch.stack(stmaps)
            data['stmap_hrs'] = torch.stack(hrs).view(-1, 1)
            data['mean_hr'] = torch.mean(data['stmap_hrs']).view(-1)
            data['video_code'] = path.split('/')[-1]
            data['fps'] = loadmat(os.path.join(p, 'gt.mat'))['gt_temp'][0][0]/loadmat(os.path.join(p, 'bpm.mat'))['bpm'][0][0]

        return data
    
@PIPELINES.register_module()
class LoadMSTmapAndSpo2(LoadMSTmap):
    def __init__(self, color_space='rgb', gt_type='hr', mode='by_frame'):
        super(LoadMSTmapAndSpo2, self).__init__(color_space, gt_type, mode)
        self.spo2_root = '/home/raid/vhwz/DATA2/vhwz/dataset/VIPL-HR/data'
        
    def load_spo2_from_file(self, path):
        with open(path, 'r') as f:
            data = f.readlines()
        data = [int(d.strip('\n')) for d in data[1:]]
        return sum(data)/len(data)
        
    def __call__(self, path):
        data =  super(LoadMSTmapAndSpo2, self).__call__(path)

        p = path.split('/')[-2].split('-')
        # print(data.keys())
        # for k, v in data.items():
        #     print(k, type(v))
        p[-1] = 'source' + p[-1][-1]
        path = os.path.join(self.spo2_root, *p, 'gt_SpO2.csv')
        data['mean_spo2'] = self.load_spo2_from_file(path)
        # hack for simple use
        data['mean_hr'] = torch.tensor(data['mean_spo2']).view(-1)
        data['stmap_hrs'] = torch.tensor([data['mean_spo2']]*len(data['stmap_hrs'])).view(-1, 1)
        return data
        
        
@PIPELINES.register_module()
class LoadPixelMap():
    def __init__(self):
        pass

    def __call__(self, path):
        data = dict()

        img_path1 = os.path.join(path, 'img_rgb.png')
        img_path2 = os.path.join(path, 'img_yuv.png')
        feature_map1 = cv2.cvtColor(cv2.imread(img_path1, cv2.COLOR_BGR2RGB))
        feature_map2 = cv2.cvtColor(cv2.imread(img_path2, cv2.COLOR_BGR2RGB))

        # if self.VerticalFlip:
        #     if random.random() < 0.5:
        #         feature_map1 = transF.vflip(feature_map1);
        #         feature_map2 = transF.vflip(feature_map2);

        # if self.transform:
        #     feature_map1 = self.transform(feature_map1)
        #     feature_map2 = self.transform(feature_map2)

        feature_map = torch.cat((feature_map1, feature_map2), dim = 0)

        # bpm_path = self.root_dir + str(dir_idx) + '/bpm.mat';
        bpm = torch.tensor(loadmat(os.path.join(path, 'bpm.mat'))['bpm'][0][0], dtype=torch.float32).view(-1)
        fps = 0
        bvp = 0

        # fps_path = self.root_dir + str(dir_idx) + '/fps.mat';
        # fps_path = os.path.join(self.paths[idx], 'fps.mat')
        # fps = sio.loadmat(fps_path)['fps'];
        # fps = fps.astype('float32');

        # bvp_path = self.root_dir + str(dir_idx) + '/bvp.mat';
        bvp_path = os.path.join(path, 'bvp.mat')
        bvp = loadmat(bvp_path)['bvp']
        bvp = bvp.astype('float32')
        bvp = bvp[0]



@PIPELINES.register_module()
class LoadStmapsFromPth():
    def __init__(self, max_seq_len=None):
        self.max_seq_len = max_seq_len

    def __call__(self, stmap_path):
        data = torch.load(stmap_path)
        data['mean_hr'] = torch.tensor(data['mean_hr'], dtype=torch.float32).view(-1)
        if 'stmap_hrs' in data.keys():
            data['stmap_hrs'] = torch.tensor(data['stmap_hrs'], dtype=torch.float32).view(-1, 1)
        else:
            L, c, h, w = data['stmaps'].shape
            data['stmap_hrs'] = data['mean_hr'].repeat(L, 1)
        if self.max_seq_len is not None:
            data['stmaps'] = data['stmaps'][0:self.max_seq_len]
            data['stmap_hrs'] = data['stmap_hrs'][0:self.max_seq_len]
        
        data['video_code'] = stmap_path.split('/')[-1].split('.')[0]

        return data


@PIPELINES.register_module()
class GenFakeStmap():
    def __init__(self, max_seq_len=30):
        self.max_seq_len = max_seq_len

    def __call__(self, stmap_path=None):
        data = {}

        stmaps = [self.synthesize_stmap(25, 300,) for _ in range(self.max_seq_len) ]
        data['stmaps'] = torch.stack([torch.tensor(s[0], dtype=torch.float32) for s in stmaps])
        data['stmap_hrs'] = torch.stack([torch.tensor(s[1], dtype=torch.float32) for s in stmaps]).view(-1, 1)
        data['mean_hr'] = torch.mean(data['stmap_hrs'], dtype=torch.float32).view(-1)

        return data

    def synthesize_stmap(self, n=25, T=300):
        res = np.zeros((3,T,n))
        t = np.arange(0, 2*3.14, 2*3.14/T)
        
        f1 = np.random.rand(1)*(4-0.7) + 0.7
        f2 = np.random.rand(1)*(0.08333-0.3333) + 0.3333
        w1 = f1*2*3.1415926
        w2 = f2*2*3.1415926
        hr = w1*60/(2*3.1415)
        P1 = np.random.rand(1)
        P2 = 1 - P1
        t1 = np.random.randint(0,T)
        t2 = np.random.randint(0,T)
        

        for i in range(n):
            for c in range(3):
                M1, M2 = np.random.rand(2)
                fi, theta = np.random.rand(2)*2*3.1415926
                #res[c,:,i] = M1*np.sin(w1*t )
                res[c,:,i] = M1*np.sin(w1*t + fi) + 0.5*M1*np.sin(2*w1*t+ fi) + M2*np.sin(w2*t + theta)
                
        res[:, :, t1:] += P1
        res[:, :, t2:] += P2
        res += 1*np.random.randn(3,T,n)
        for c in range(3):
            for i in range(n):
                res[c,:,i] = self.minmaxscale(res[c,:,i])
        return res, float(hr)

    def minmaxscale(self, s):
        M = s.max()
        m  = s.min()
        s = (s - m)/(M - m)
        return s


@PIPELINES.register_module()
class LoadStmapsFromImage():
    def __init__(self, ):
        pass

    def __call__(self, ):
        pass