from ..builder import PIPELINES
import torch
import torchvision

@PIPELINES.register_module()
class ScaleHR():
    '''
    inputs: stmaps: tensor(b, c, h, w), range(0, 1)
    '''
    def __init__(self, min=42, max=240):
        self.min = min
        self.max = max

    def __call__(self, data):
        if 'hrs' in data.keys():
            hrs = data['hrs']
            hrs = (hrs - self.min)/(self.max - self.min)
            data['hrs'] = hrs

        if 'stmap_hrs' in data.keys():
            hrs = data['stmap_hrs']
            hrs = (hrs - self.min)/(self.max - self.min)
            data['stmap_hrs'] = hrs
        return data


@PIPELINES.register_module()
class NormHR():
    '''
    inputs: stmaps: tensor(b, c, h, w), range(0, 1)
    '''
    def __init__(self):
        pass

    def __call__(self, data):

        if 'stmap_hrs' in data.keys():
            hrs = data['stmap_hrs']
            hrs = hrs/6
            data['stmap_hrs'] = hrs
        data['mean_hr'] /= 6

        return data