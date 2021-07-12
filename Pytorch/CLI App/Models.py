import re
import torch
from torch import nn, optim
from torchvision import models, transforms, ops


class Model(nn.Module):
    def __init__(self, modeltype="classifier"):
        super(Model, self).__init__()

        self.modeltype = modeltype

        if re.match(r"classifier", self.modeltype, re.IGNORECASE):
            self.model = models.resnet18(pretrained=True, progress=True)
        elif re.match(r"detector", self.modeltype, re.IGNORECASE):
            self.model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True, progress=True)
        elif re.match(r"segmentor", self.modeltype, re.IGNORECASE):
            self.model = models.segmentation.lraspp_mobilenet_v3_large(pretrained=True, progress=True)
    
    def forward(self, x):
        return self.model(x)

