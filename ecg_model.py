import torch
from torch import nn
from torchvision.models import resnet18
import torch.nn.functional as F

class ECGResnet18(nn.Module):
    def __init__(self, opt):
        super(ECGResnet18, self).__init__()
        self.res = resnet18(pretrained=False) 
        clf = nn.Sequential(nn.Linear(512, opt.hsz1),
                            nn.ReLU(),
                            nn.Linear(opt.hsz1, opt.outsz)) #hidden size 1, output size
        self.res.fc = clf

    def forward(self,spec):
        return self.res(spec)
