import torch
import torch.nn as nn
import unittest


class CustomMSELoss(nn.Module):
    def __init__(self, lower_bound=0.3, upper_bound=0.7):
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def forward(self, input, target):
        # Create a mask where values are outside the acceptable ranges
        mask = ~(((target == 0) & (input <= self.lower_bound)) | 
                 ((target == 1) & (input >= self.upper_bound)))
        
        # Compute standard MSE only for masked (unacceptable) values
        loss = torch.mean((mask * (input - target)) ** 2)
        return loss
    


class TestCustomMSELoss(unittest.TestCase):
    def setUp(self):
        self.loss_fn = CustomMSELoss(lower_bound=0.3, upper_bound=0.7)
    
    def test_loss_within_bounds(self):
        input_tensor = torch.tensor([0.2, 0.8, 0.5, 0.6], dtype=torch.float32)
        target_tensor = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
        loss = self.loss_fn(input_tensor, target_tensor)
        self.assertGreater(loss.item(), 0, "Loss should be greater than 0 for out-of-bound values")
    
    def test_loss_zero_when_within_bounds(self):
        input_tensor = torch.tensor([0.3, 0.7, 0.2, 0.8], dtype=torch.float32)
        target_tensor = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
        loss = self.loss_fn(input_tensor, target_tensor)
        self.assertEqual(loss.item(), 0.0, "Loss should be 0 when values are within acceptable bounds")
    
    def test_loss_all_zeros(self):
        input_tensor = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        target_tensor = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
        loss = self.loss_fn(input_tensor, target_tensor)
        self.assertEqual(loss.item(), 0.0, "Loss should be 0 when all inputs match target zero case")
    
    def test_loss_all_ones(self):
        input_tensor = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        target_tensor = torch.tensor([1, 1, 1, 1], dtype=torch.float32)
        loss = self.loss_fn(input_tensor, target_tensor)
        self.assertEqual(loss.item(), 0.0, "Loss should be 0 when all inputs match target one case")
    
    def test_loss_mixed_values(self):
        input_tensor = torch.tensor([0.1, 0.9, 0.4, 0.6], dtype=torch.float32)
        target_tensor = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
        loss = self.loss_fn(input_tensor, target_tensor)
        self.assertGreater(loss.item(), 0, "Loss should be greater than 0 for mixed cases")

if __name__ == "__main__":
    unittest.main()
