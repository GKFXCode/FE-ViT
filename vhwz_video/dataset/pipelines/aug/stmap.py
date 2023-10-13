from ...builder import PIPELINES
import torch

@PIPELINES.register_module()
class AddGaussionNoise():
    '''
    inputs: stmaps: tensor(b, c, h, w), range(0, 1)
    '''
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, data):
        stmaps = data['stmaps']
        b, c, h, w = stmaps.shape
        end = int(b*self.prob)
        if end == 0: end = 1
        idx = torch.randperm(b)[0:end]

        noise = torch.randn(end,c,h,w)
        stmaps[idx] += noise
        data['stmaps'] = stmaps

        return data