import torch
import torch.nn as nn

class NormalizeInput(torch.nn.Module):
    def __init__(self, mean, std, channels) :
        super(NormalizeInput, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.channels = channels
        
    def forward(self, input): 
        device = input.device
        mean = self.mean.reshape(1, self.channels, 1, 1).to(device)
        std = self.std.reshape(1, self.channels, 1, 1).to(device)
        norm_img = (input - mean) / std
        return norm_img