import torch
from torch import nn

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, x1, x2, y):
        squared_distance = torch.sum(torch.pow(x1-x2, 2), dim=1)
        sqrt_distance = torch.sqrt(squared_distance)
        margin_distance = self.margin - sqrt_distance
        sqrt_distance = torch.clamp(margin_distance, min=0.0)
        
        loss = (y*squared_distance) + ((1-y)*torch.pow(sqrt_distance, 2))
        loss = torch.sum(loss, dim=0) / loss.shape[0]
        return loss


############### Test ####################
x1 = torch.rand(256, 2048)
x2 = torch.rand(256, 2048)
y  = torch.randint(0, 2, (256, ))

criterion = ContrastiveLoss(margin=1.0)
loss = criterion(x1, x2, y)

print(loss)
#########################################
