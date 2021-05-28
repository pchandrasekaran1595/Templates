import torch
from torch import nn, optim
from torch.nn.utils import weight_norm as WN
import torch.nn.functional as F

class Model_Name(nn.Module):
    def __init__(self, use_DP=True, DP=0.5, *args, **kwargs, ):

        super(Model_Name, self).__init__()
        
        self.use_DP = use_DP
        self.DP_ = nn.Dropout(p=DP) # nn.Dropout2d() or nn.Dropout3d()

    def getOptimizer(self, lr=1e-3, wd=0, *args, **kwargs):
        return optim.____

    def get________LR(self, optimizer=None, *args, **kwargs):
        return optim.lr_scheduler.________(optimizer=optimizer, )

    def forward(self, x):
        if self.use_DP:
            
            return x        
        else:
            
            return x
