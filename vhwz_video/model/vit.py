from .build import MODEL
import torch.nn as nn
import timm
import torch.nn.functional as F 
from timm.models.vision_transformer import VisionTransformer as timmVisionTransformer
from .heads import FCActHead

@MODEL.register('Vit')
class VisionTransformer(nn.Module):
    def __init__(self, cfg):
        super(VisionTransformer, self).__init__()
        self.vit = timmVisionTransformer(img_size=(300, 189), in_chans=1, depth=6)
        self.head = FCActHead('relu')

    def forward(self, stmaps):
        x = stmaps.unsqueeze(1)
        y = self.vit(x)
        y = self.head(y)
        return y
    
    def loss_function(self, pred, gt):
        return F.mse_loss(pred, gt)
    
@MODEL.register('ImageVit')
class ImageVisionTransformer(nn.Module):
    def __init__(self, cfg):
        super(VisionTransformer, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.head = FCActHead('relu')

    def forward(self, stmaps):
        y = self.vit(stmaps)
        y = self.head(y)
        return y
    
    def loss_function(self, pred, gt):
        return F.mse_loss(pred, gt)