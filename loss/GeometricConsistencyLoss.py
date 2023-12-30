
import torch.nn as nn
import torch.nn.functional as F

########################################################################################
# Geometric Consistency Loss
# Maintains geometric integrity of depth information.
# This will help in preserving the contours and shapes of letters in the displacement maps.
########################################################################################
class GeometricConsistencyLoss(nn.Module):
    def __init__(self):
        super(GeometricConsistencyLoss, self).__init__()

    def forward(self, predicted_map, target_map):
        # Calculate gradients in x and y direction
        # These gradients represent the change in depth (or displacement) across pixels
        grad_x_pred, grad_y_pred = self.compute_gradients(predicted_map)
        grad_x_target, grad_y_target = self.compute_gradients(target_map)

        # Calculate the loss as the mean squared error between the gradients of the predicted and target maps
        loss_x = F.mse_loss(grad_x_pred, grad_x_target)
        loss_y = F.mse_loss(grad_y_pred, grad_y_target)

        # Combine the losses
        loss = loss_x + loss_y

        return loss

    def compute_gradients(self, map):
        # Function to compute gradients in the x and y direction
        grad_x = map[:, :, :, :-1] - map[:, :, :, 1:]
        grad_y = map[:, :, :-1, :] - map[:, :, 1:, :]

        return grad_x, grad_y

