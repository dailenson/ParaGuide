import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class RN18_Encoder(nn.Module):
    def __init__(self, pretrained=False):
        super(RN18_Encoder, self).__init__()
        self.is_pretrained = pretrained
        self.backbone = torchvision.models.resnet18(pretrained=self.is_pretrained)

        # to get feature map -- model without 'avgpool' and 'fc'
        self.features = nn.Sequential()
        for name, module in self.backbone.named_children():
            if name not in ['avgpool', 'fc']:
                self.features.add_module(name, module)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, pool=True):
        x = self.features(x)
        if pool:
            x = self.gap(x)
            x = torch.flatten(x, 1)
        return x