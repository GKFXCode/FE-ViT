import torch.nn as nn


func = {'sigmoid':nn.Sigmoid(), 'softmax':nn.Softmax(dim=0), 'relu':nn.ReLU(), 'none':nn.Identity()}

class FCActHead(nn.Module):
    def __init__(self, act_func='sigmoid'):
        super(FCActHead, self).__init__()
        self.fc_regression = nn.Sequential(
            nn.Linear(1000, 1),
            func[act_func]
        )

    def forward(self, x):
        x = self.fc_regression(x)
        return x