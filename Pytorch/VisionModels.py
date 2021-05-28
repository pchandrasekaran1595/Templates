# ********************************************************************************************* #

import torch.nn.functional as F
from torch import nn, optim
from torchvision import models
import re

# ********************************************************************************************* #

class Classifiers(nn.Module):
    def __init__(self, name=None, pretrained=False, in_channels=3, OL=None, retrain_base=False):

        super(Classifiers, self).__init__()

        if re.match(r"\balexnet\b", name, flags=re.IGNORECASE):
            self.model = models.alexnet(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()

        elif re.match(r"\bresnet18\b", name, flags=re.IGNORECASE):
            self.model = models.resnet18(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()
        elif re.match(r"\bresnet34\b", name, flags=re.IGNORECASE):
            self.model = models.resnet34(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()
        elif re.match(r"\bresnet50\b", name, flags=re.IGNORECASE):
            self.model = models.resnet50(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()
        elif re.match(r"\bresnet101\b", name, flags=re.IGNORECASE):
            self.model = models.resnet101(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()
        elif re.match(r"\bresnet152\b", name, flags=re.IGNORECASE):
            self.model = models.resnet152(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()
        elif re.match(r"\bresnext50_32x4d\b", name, flags=re.IGNORECASE):
            self.model = models.resnext50_32x4d(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()
        elif re.match(r"\bresnext101_32x8d\b", name, flags=re.IGNORECASE):
            self.model = models.resnext101_32x8d(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()
        elif re.match(r"\bwide_resnet50_2\b", name, flags=re.IGNORECASE):
            self.model = models.wide_resnet50_2(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()
        elif re.match(r"\bwide_resnet101_2\b", name, flags=re.IGNORECASE):
            self.model = models.wide_resnet101_2(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()

        elif re.match(r"\bvgg11\b", name, flags=re.IGNORECASE):
            self.model = models.vgg11(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()
        elif re.match(r"\bvgg11_bn\b", name, flags=re.IGNORECASE):
            self.model = models.vgg11_bn(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()
        elif re.match(r"\bvgg13\b", name, flags=re.IGNORECASE):
            self.model = models.vgg13(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()
        elif re.match(r"\bvgg13_bn\b", name, flags=re.IGNORECASE):
            self.model = models.vgg13_bn(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()
        elif re.match(r"\bvgg16\b", name, flags=re.IGNORECASE):
            self.model = models.vgg16(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()
        elif re.match(r"\bvgg16_bn\b", name, flags=re.IGNORECASE):
            self.model = models.vgg16_bn(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()
        elif re.match(r"\bvgg19\b", name, flags=re.IGNORECASE):
            self.model = models.vgg19(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()
        elif re.match(r"\bvgg19_bn\b", name, flags=re.IGNORECASE):
            self.model = models.vgg19_bn(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()

        elif re.match(r"\bsqueezenet1_0\b", name, flags=re.IGNORECASE):
            self.model = models.squeezenet1_0(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()
        elif re.match(r"\bsqueezenet1_0\b", name, flags=re.IGNORECASE):
            self.model = models.squeezenet1_0(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()

        elif re.match(r"\bdensenet121\b", name, flags=re.IGNORECASE):
            self.model = models.densenet121(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()
        elif re.match(r"\bdensenet161\b", name, flags=re.IGNORECASE):
            self.model = models.densenet161(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()
        elif re.match(r"\bdensenet169\b", name, flags=re.IGNORECASE):
            self.model = models.densenet169(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()
        elif re.match(r"\bdensenet201\b", name, flags=re.IGNORECASE):
            self.model = models.densenet201(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()

        elif re.match(r"\bmobilenet_v2\b", name, flags=re.IGNORECASE):
            self.model = models.mobilenet_v2(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()
        elif re.match(r"\bmobilenet_v3_small\b", name, flags=re.IGNORECASE):
            self.model = models.mobilenet_v3_small(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()

        elif re.match(r"\bmnasnet0_5\b", name, re.IGNORECASE):
            self.model = models.mnasnet0_5(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()
        elif re.match(r"\bmnasnet0_75\b", name, re.IGNORECASE):
            self.model = models.mnasnet0_75(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()
        elif re.match(r"\bmnasnet1_0\b", name, re.IGNORECASE):
            self.model = models.mnasnet1_0(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()
        elif re.match(r"\bmnasnet1_3\b", name, re.IGNORECASE):
            self.model = models.mnasnet1_3(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()

        if re.match(r"\bresnet\w", name, flags=re.IGNORECASE) or re.match(r"\bresnext\w", name, flags=re.IGNORECASE) or \
    re.match(r"\bwide\w", name, flags=re.IGNORECASE):
            if in_channels != 3 or (in_channels == 3 and retrain_base):
                self.model.conv1 = nn.Conv2d(in_channels=in_channels,
                                            out_channels=self.model.conv1.out_channels,
                                            kernel_size=self.model.conv1.kernel_size,
                                            stride=self.model.conv1.stride,
                                            padding=self.model.conv1.padding)

            if OL is not None:
                self.model.fc = nn.Linear(in_features=self.model.fc.in_features,
                                          out_features=OL)

        if re.match(r"\bvgg\w", name, flags=re.IGNORECASE) or re.match(r"\balexnet\b", name, flags=re.IGNORECASE):
            if in_channels != 3 or (in_channels == 3 and retrain_base):
                self.model.features[0] = nn.Conv2d(in_channels=in_channels,
                                                out_channels=self.model.features[0].out_channels,
                                                kernel_size=self.model.features[0].kernel_size,
                                                stride=self.model.features[0].stride,
                                                padding=self.model.features[0].padding)
            
            if OL is not None:
                self.model.classifier[-1] = nn.Linear(in_features=self.model.classifier[-1].in_features,
                                                    out_features=OL)

        if re.match(r"\bsq\w", name, flags=re.IGNORECASE):
            if in_channels != 3 or (in_channels == 3 and retrain_base):
                self.model.features[0] = nn.Conv2d(in_channels=in_channels,
                                                out_channels=self.model.features[0].out_channels,
                                                kernel_size=self.model.features[0].kernel_size,
                                                stride=self.model.features[0].stride,
                                                padding=self.model.features[0].padding)

            if OL is not None:
                self.model.classifier[1] = nn.Conv2d(in_channels=self.model.classifier[1].in_channels,
                                                    out_channels=OL,
                                                    kernel_size=self.model.classifier[1].kernel_size,
                                                    stride=self.model.classifier[1].stride,
                                                    padding=self.model.classifier[1].padding)

        if re.match(r"\bden\w", name, flags=re.IGNORECASE):
            if in_channels != 3 or (in_channels == 3 and retrain_base):
                self.model.features.conv0 = nn.Conv2d(in_channels=in_channels,
                                                    out_channels=self.model.features.conv0.out_channels,
                                                    kernel_size=self.model.features.conv0.kernel_size,
                                                    stride=self.model.features.conv0.stride,
                                                    padding=self.model.features.conv0.padding)

            if OL is not None:
                self.model.classifier = nn.Linear(in_features=self.model.classifier.in_features,
                                                out_features=OL)
            
        if re.match(r"\bmobilenet\w", name, flags=re.IGNORECASE):
            if in_channels != 3 or (in_channels == 3 and retrain_base):
                self.model.features[0][0] = nn.Conv2d(in_channels=in_channels,
                                                    out_channels=self.model.features[0][0].out_channels,
                                                    kernel_size=self.model.features[0][0].kernel_size,
                                                    stride=self.model.features[0][0].stride,
                                                    padding=self.model.features[0][0].padding)

            if OL is not None:
                self.model.classifier[-1] = nn.Linear(in_features=self.model.classifier[-1].in_features,
                                                    out_features=OL)

        if re.match(r"\bmnas\w", name, flags=re.IGNORECASE):
            if in_channels != 3 or (in_channels == 3 and retrain_base):
                self.model.layers[0] = nn.Conv2d(in_channels=in_channels,
                                                    out_channels=self.model.layers[0].out_channels,
                                                    kernel_size=self.model.layers[0].kernel_size,
                                                    stride=self.model.layers[0].stride,
                                                    padding=self.model.layers[0].padding)
                
            if OL is not None:
                self.model.classifier[-1] = nn.Linear(in_features=self.model.classifier[-1].in_features,
                                                    out_features=OL)

    def freeze(self):
        for params in self.parameters():
            params.requires_grad = False

    def getOptimizer(self, A_S=True, lr=1e-3, wd=0):
        if A_S:
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        else:
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

    def getStepLR(self, optimizer=None, step_size=5, gamma=0.1):
        return optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)

    def getMultiStepLR(self, optimizer=None, milestones=None, gamma=0.1):
        return optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=gamma)

    def getPlateauLR(self, optimizer=None, patience=5, eps=1e-8):
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=patience, eps=eps, verbose=True)

    def forward(self, x):
        x = self.model(x)
        # x = F.log_softmax(self.model(x), dim=1)
        return x

# ********************************************************************************************* #

class FeatureExtractors(nn.Module):
    def __init__(self, name=None):

        super(FeatureExtractors, self).__init__()

        if re.match(r"\balexnet\b", name, flags=re.IGNORECASE):
            self.model = models.alexnet(pretrained=True, progress=True)
        elif re.match(r"\bresnet18\b", name, flags=re.IGNORECASE):
            self.model = models.resnet18(pretrained=True, progress=True)      
        elif re.match(r"\bresnet34\b", name, flags=re.IGNORECASE):
            self.model = models.resnet34(pretrained=True, progress=True)
        elif re.match(r"\bresnet50\b", name, flags=re.IGNORECASE):
            self.model = models.resnet50(pretrained=True, progress=True)
        elif re.match(r"\bresnet101\b", name, flags=re.IGNORECASE):
            self.model = models.resnet101(pretrained=True, progress=True)
        elif re.match(r"\bresnet152\b", name, flags=re.IGNORECASE):
            self.model = models.resnet152(pretrained=True, progress=True)
        elif re.match(r"\bresnext50_32x4d\b", name, flags=re.IGNORECASE):
            self.model = models.resnext50_32x4d(pretrained=True, progress=True)
        elif re.match(r"\bresnext101_32x8d\b", name, flags=re.IGNORECASE):
            self.model = models.resnext101_32x8d(pretrained=True, progress=True)
        elif re.match(r"\bwide_resnet50_2\b", name, flags=re.IGNORECASE):
            self.model = models.wide_resnet50_2(pretrained=True, progress=True)
        elif re.match(r"\bwide_resnet101_2\b", name, flags=re.IGNORECASE):
            self.model = models.wide_resnet101_2(pretrained=True, progress=True)
        elif re.match(r"\bvgg11\b", name, flags=re.IGNORECASE):
            self.model = models.vgg11(pretrained=True, progress=True)
        elif re.match(r"\bvgg11_bn\b", name, flags=re.IGNORECASE):
            self.model = models.vgg11_bn(pretrained=True, progress=True)
        elif re.match(r"\bvgg13\b", name, flags=re.IGNORECASE):
            self.model = models.vgg13(pretrained=True, progress=True)
        elif re.match(r"\bvgg13_bn\b", name, flags=re.IGNORECASE):
            self.model = models.vgg13_bn(pretrained=True, progress=True)
        elif re.match(r"\bvgg16\b", name, flags=re.IGNORECASE):
            self.model = models.vgg16(pretrained=True, progress=True)
        elif re.match(r"\bvgg16_bn\b", name, flags=re.IGNORECASE):
            self.model = models.vgg16_bn(pretrained=True, progress=True)
        elif re.match(r"\bvgg19\b", name, flags=re.IGNORECASE):
            self.model = models.vgg19(pretrained=True, progress=True)
        elif re.match(r"\bvgg19_bn\b", name, flags=re.IGNORECASE):
            self.model = models.vgg19_bn(pretrained=True, progress=True)
        elif re.match(r"\bsqueezenet1_0\b", name, flags=re.IGNORECASE):
            self.model = models.squeezenet1_0(pretrained=True, progress=True)
        elif re.match(r"\bsqueezenet1_0\b", name, flags=re.IGNORECASE):
            self.model = models.squeezenet1_0(pretrained=True, progress=True)
        elif re.match(r"\bdensenet121\b", name, flags=re.IGNORECASE):
            self.model = models.densenet121(pretrained=True, progress=True)
        elif re.match(r"\bdensenet161\b", name, flags=re.IGNORECASE):
            self.model = models.densenet161(pretrained=True, progress=True)
        elif re.match(r"\bdensenet169\b", name, flags=re.IGNORECASE):
            self.model = models.densenet169(pretrained=True, progress=True)
        elif re.match(r"\bdensenet201\b", name, flags=re.IGNORECASE):
            self.model = models.densenet201(pretrained=True, progress=True)
        elif re.match(r"\bmobilenet_v2\b", name, flags=re.IGNORECASE):
            self.model = models.mobilenet_v2(pretrained=True, progress=True)
        elif re.match(r"\bmobilenet_v3_small\b", name, flags=re.IGNORECASE):
            self.model = models.mobilenet_v3_small(pretrained=True, progress=True)
        elif re.match(r"\bmnasnet0_5\b", name, re.IGNORECASE):
            self.model = models.mnasnet0_5(pretrained=True, progress=True)
        elif re.match(r"\bmnasnet0_75\b", name, re.IGNORECASE):
            self.model = models.mnasnet0_75(pretrained=True, progress=True)
        elif re.match(r"\bmnasnet1_0\b", name, re.IGNORECASE):
            self.model = models.mnasnet1_0(pretrained=True, progress=True)
        elif re.match(r"\bmnasnet1_3\b", name, re.IGNORECASE):
            self.model = models.mnasnet1_3(pretrained=True, progress=True)

        self.freeze()
        self.model = nn.Sequential(*[*self.model.children()][:-1])

    def freeze(self):
        for params in self.parameters():
            params.requires_grad = False

    def forward(self, x):
        return self.model(x)
    
# ********************************************************************************************* #

class Segmenters(nn.Module):
    def __init__(self, name="lraspp_mobilenet_v3_large"):
        super(Segmenters, self).__init__()

        if re.match(r"\bfcn_resnet50\b", name, flags=re.IGNORECASE):
            self.model = models.segmentation.fcn_resnet50(pretrained=True, progress=True)
        elif re.match(r"\bfcn_resnet101\b", name, flags=re.IGNORECASE):
            self.model = models.segmentation.fcn_resnet101(pretrained=True, progress=True)
        elif re.match(r"\bdeeplabv3_resnet50\b", name, flags=re.IGNORECASE):
            self.model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
        elif re.match(r"\bdeeplabv3_resnet101\b", name, flags=re.IGNORECASE):
            self.model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
        elif re.match(r"\bdeeplabv3_mobilenet_v3_large\b", name, flags=re.IGNORECASE):
            self.model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True, progress=True)
        elif re.match(r"\blraspp_mobilenet_v3_large\b", name, flags=re.IGNORECASE):
            self.model = models.segmentation.lraspp_mobilenet_v3_large(pretrained=True, progress=True)

    def forward(self, x):
        return self.model(x)
    
# ********************************************************************************************* #

class Detectors(nn.Module):
    def __init__(self, name="FRCNN Large 320"):
        super(Detectors, self).__init__()

        if re.match(r"MV3 Large 320", name, re.IGNORECASE):
            self.model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
        elif re.match(r"MV3 Large", name, re.IGNORECASE):
            self.model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
        elif re.match(r"ResNet", name, re.IGNORECASE):
            self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        elif re.match(r"Retina", name, re.IGNORECASE):
            self.model = models.detection.retinanet_resnet50_fpn(pretrained=True)
        elif re.match(r"Mask", name, re.IGNORECASE):
            self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        elif re.match(r"Keypoint", name, re.IGNORECASE):
            self.model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    
    def forward(self, x):
        return self.model(x)
 
# ********************************************************************************************* #
