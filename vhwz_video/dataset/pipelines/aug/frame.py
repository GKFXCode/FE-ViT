from ...builder import PIPELINES
import torch

@PIPELINES.register_module
class RectifyIllumination():
    '''
    for frame
    '''
    def __init__(self) -> None:
        pass

    def __call__(self, data):
        stmaps = data['stmaps']


        return data