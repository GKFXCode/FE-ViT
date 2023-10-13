import numpy as np
from scipy.special.orthogonal import hermite
import scipy
import torch
from ..builder import PIPELINES
import torchvision
from einops import rearrange
from scipy import signal
from scipy.signal import firwin, filtfilt
from scipy.signal import butter, lfilter

@PIPELINES.register_module()
class RandomMask():
    '''
    inputs: stmaps: tensor(b, c, h, w), range(0, 1)
            prob: prob
            low, high: lowest and highest length of mask 
    '''
    def __init__(self, prob=0.5, low=10, high=30) -> None:
        self.prob = prob
        self.low = low
        self.high = high

    def __call__(self, data):
        stmaps = data['stmaps']
        if len(stmaps.shape) == 4:
            'stmaps.shape = (b, 3, 300, 25)'
            b, c, h, w = stmaps.shape
            idx = torch.randperm(b)
            end = int(b*self.prob)
            if end == 0: 
                end = 1

            for i in idx[0:end]:  # mask in 50% of data
                s = np.random.randint(0, h)
                L = np.random.randint(low=self.low,high=self.high)
                stmaps[i,:, s:s+L,:] = 0
        elif len(stmaps.shape) == 3:
            'stmaps.shape = (3, 63, 300)'
            c, w, h = stmaps.shape
            if torch.randn(1) > 0:
                s = np.random.randint(0, h)
                L = np.random.randint(low=self.low,high=self.high)
                stmaps[:, :, s:s+L] = 0
                
        return data

@PIPELINES.register_module()
class Resize():
    '''
    inputs: stmaps: tensor(b, c, h, w), range(0, 1)
    '''
    def __init__(self, size):
        self.size = size
        self._resize = torchvision.transforms.Resize(size)

    def __call__(self, data):
        data['stmaps'] = self._resize(data['stmaps'])
        return data

@PIPELINES.register_module()
class Detrend():
    '''
    inputs: stmaps: tensor(b, c, h, w), range(0, 1)
    '''
    def __init__(self, scale=None):
        self.scale = scale

    def __call__(self, data):
        z = np.array(data['stmaps'], dtype=float)
        if len(z.shape) == 4:
            b, c, h ,w = z.shape
            z = rearrange(z, 'b c h w -> h(b c w)') 
            y = self.detrending(z)
            if self.scale:
                y = self._scale(y)
            y = rearrange(y, ' h(b c w)-> b c h w', b=b,c=c,h=h,w=w)

        elif len(z.shape) == 3:
            c, h, w = z.shape
            z = rearrange(z, 'c h w -> h(c w)') 
            y = self.detrending(z)
            if self.scale:
                y = self._scale(y)
            y = rearrange(y, ' h(c w)-> c h w', c=c,h=h,w=w)
        # data['stmaps'] = scipy.signal.detrend(data['stmaps'], axis=2)
        data['stmaps'] = torch.tensor(y, dtype=torch.float32)
        return data

    def detrending(self, z):
        '''
        input: signal np.array(signal_len, batch_size)
        output: detrended signal np.array(signal_len, batch_size)
        '''
        T = z.shape[0]
        lamb = 10
        I = np.eye(T)
        D2 = (np.diag(np.ones((T, 1)).reshape(-1), 0)[0:T-2, 0:T] + 
            np.diag(np.ones((T, 1)).reshape(-1)*-2, 1)[0:T-2, 0:T] + 
            np.diag(np.ones((T, 1)).reshape(-1), 2)[0:T-2, 0:T])

        z = (I - np.linalg.inv(I + (lamb**2)*(D2.T@D2)))@z

        return z

    def _scale(self, x):
        '''
        input: x np.array(seq_len, batch_size)
        output : scaled x
        '''
        M = np.max(x, axis=1).reshape(-1,1)
        m = np.min(x, axis=1).reshape(-1,1)
        return (x-m)/(M-m)

@PIPELINES.register_module()
class MovingAverage():
    '''
    inputs: stmaps: tensor(b, c, h, w), range(0, 1)
    '''
    def __init__(self):
        pass

    def __call__(self, data):
        z = np.array(data['stmaps'], dtype=float)
        b, c, h ,w = z.shape
        z = rearrange(z, 'b c h w -> h(b c w)') 
        y = self.ma(z)

        y = rearrange(y, ' h(b c w)-> b c h w', b=b,c=c,h=h,w=w)
        data['stmaps'] = torch.tensor(y, dtype=torch.float32)
        return data

    def ma(self, x, L=3):
        seqlen, batch = x.shape
        #L-point filter
        b = (np.ones((L)))/L
        y = np.zeros(x.shape)
        for i in range(batch):
            y[..., i] = signal.convolve(x[..., i], b, mode='same')
        return y

@PIPELINES.register_module()
class Bandpass():
    '''
    inputs: stmaps: tensor(b, c, h, w), range(0, 1)
    '''
    def __init__(self, framerate=25, order=5, lowcut=0.7, highcut=4):
        self.b, self.a = self._build_butter_bandpass(lowcut, highcut, framerate, order=order)

    def __call__(self, data):
        x = np.array(data['stmaps'], dtype=float)
        b, c, h ,w = x.shape
        x = rearrange(x, 'b c h w -> h(b c w)') 
        y = self.butter_bandpass_filter(x)
        y = rearrange(y, ' h(b c w)-> b c h w', b=b,c=c,h=h,w=w)
        data['stmaps'] = torch.tensor(y, dtype=torch.float32)
        return data

    def _build_butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, x):
        '''
        x: np.array(seqLen, batch)
        '''
        seqLen, batch = x.shape
        y = np.zeros(x.shape)
        for i in range(batch):
            y[..., i] = lfilter(self.b, self.a, x[..., i])
        return y



@PIPELINES.register_module()    
class Standardize():
    '''
    inputs: stmaps: tensor(b, c, h, w), range(0, 1)
    '''
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, data):
        stmaps = data['stmaps']
        data['stmaps'] = (stmaps - self.mean)/self.std
        return data

@PIPELINES.register_module()
class AddSaltPepperNoise():
    '''
    inputs: stmaps: tensor(b, c, h, w), range(0, 1)
    '''
    def __init__(self, prob=0.1, threshold=0.1, pepper_val=1):
        self.prob = prob
        self.threshold = threshold
        self.pepper_val = pepper_val

    def __call__(self, data):
        stmaps = data['stmaps']
        if len(stmaps.shape) == 4:
            b, c, h, w = stmaps.shape
            end = int(b*self.prob)
            if end == 0: end = 1
            idx = torch.randperm(b)[0:end]

            noise = torch.rand(end,c,h,w)
            stmaps[idx][noise < self.threshold/2] = 0
            stmaps[idx][noise > 1- self.threshold/2] = self.pepper_val
        elif len(stmaps.shape) == 3:
            c, h, w = stmaps.shape
            if torch.rand(1) < self.prob:
                noise = torch.rand(c,h,w)
                stmaps[noise < self.threshold/2] = 0
                stmaps[noise > 1- self.threshold/2] = self.pepper_val

        data['stmaps'] = stmaps
        return data

@PIPELINES.register_module()
class AddGaussianNoise():
    '''
    inputs: stmaps: tensor(b, c, h, w), range(0, 1)
    '''
    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, data):
        stmaps = data['stmaps']
        if torch.rand(1) < self.prob:
            stmaps += torch.randn(stmaps.shape)
            stmaps[stmaps<0] = 0
            stmaps[stmaps>1] = 1

        data['stmaps'] = stmaps
        return data

