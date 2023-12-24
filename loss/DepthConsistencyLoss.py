import torch

import torch.nn as nn

########################################################################################
# DepthConsistencyLoss
# This is a robust loss that combines the benefits of L1 and L2 losses.
# It can be particularly useful if there's a lot of noise in the damaged maps.
# Depth data is uncertain in some places
# This loss can help in smoothing the depth map without losing essential details.
########################################################################################
class DepthConsistencyLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(DepthConsistencyLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, generated_depth, target_depth):
        # Assuming generated_depth and target_depth are tensors representing depth maps
        # Charbonnier Loss: sqrt((x - y)^2 + epsilon)
        loss = torch.mean(torch.sqrt((generated_depth - target_depth) ** 2 + self.epsilon))

        return loss