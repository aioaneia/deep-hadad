import torch
import unittest

import models.DHadadLossFunctions as LossFunctions


class TestLossFunctions(unittest.TestCase):
    def setUp(self):
        self.epsilon = 1e-6  # Small number to assert equality
        self.input = torch.rand(1, 1, 256, 256, requires_grad=True)
        self.target = torch.rand(1, 1, 256, 256)
        self.zero_target = torch.zeros_like(self.input)

    def test_l1_loss_identical(self):
        loss = LossFunctions.DHadadLossFunctions.l1_loss(self.input, self.input)
        self.assertAlmostEqual(loss.item(), 0, delta=self.epsilon, msg="L1 Loss should be zero for identical inputs")

    def test_l1_loss_zeros(self):
        loss = LossFunctions.DHadadLossFunctions.l1_loss(self.input, self.zero_target)
        expected_loss = self.input.mean().item()  # Mean of absolute differences (input and zero)
        self.assertAlmostEqual(loss.item(), expected_loss, delta=self.epsilon, msg="L1 Loss fails with zero target")

    def test_l1_loss_non_identical(self):
        loss = LossFunctions.DHadadLossFunctions.l1_loss(self.input, self.target)
        manual_loss = torch.abs(self.input - self.target).mean().item()
        self.assertAlmostEqual(loss.item(), manual_loss, delta=self.epsilon,
                               msg="L1 Loss calculation does not match manual calculation")

    def test_l1_loss_negative_case(self):
        negative_input = -self.input  # Creating a negative case
        loss = LossFunctions.DHadadLossFunctions.l1_loss(negative_input, self.zero_target)
        expected_loss = negative_input.abs().mean().item()
        self.assertAlmostEqual(loss.item(), expected_loss, delta=self.epsilon, msg="L1 Loss fails with negative inputs")

    def test_ssim_loss(self):
        loss = LossFunctions.DHadadLossFunctions.ssim_loss(self.input, self.input)

        self.assertAlmostEqual(loss.item(), 0, delta=self.epsilon)

    def test_edge_loss(self):
        edge_target = self.target.clone()
        edge_target[:, :, :128, :] = 0  # Half the target is zeroed out
        loss = LossFunctions.DHadadLossFunctions.edge_loss(self.input, edge_target)

        self.assertGreater(loss.item(), 0)


if __name__ == '__main__':
    unittest.main()
