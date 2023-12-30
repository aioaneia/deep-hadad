
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
SharpnessLoss 
"""
class SharpnessLoss(nn.Module):
    def __init__(self):
        super(SharpnessLoss, self).__init__()

        self.kernel = torch.tensor([[-1, -1, -1],
                                    [-1,  9, -1],
                                    [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def forward(self, input, target):
        self.kernel  = self.kernel.to(input.device)
        sharp_input  = F.conv2d(input, self.kernel, padding=1)
        sharp_target = F.conv2d(target, self.kernel, padding=1)
        
        return F.mse_loss(sharp_input, sharp_target)
