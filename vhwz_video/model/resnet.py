import torch.nn as nn
import torch.nn.functional as F
from .build import MODEL
import torchvision

@MODEL.register('Resnet')
class VisionTransformer(nn.Module):
    def __init__(self, cfg):
        super(VisionTransformer, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True, progress=False)
        self.fc_regression = nn.Sequential(
            nn.Linear(1000, 1),
            nn.ReLU()
        )

    def forward(self, stmaps):
        out = self.resnet(stmaps)
        out = self.fc_regression(out)
        return out
    
    def loss_function(self, pred, gt):
        return F.mse_loss(pred, gt)