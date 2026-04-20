# Source: https://github.com/uiuc-arc/CGT/tree/main
# Code authors: Rem Yang, Jacob Laurel, Sasa Misailovic, and Gagandeep Singh.

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import random

import sys
sys.path.append('../')
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

means = (0.4914, 0.4822, 0.4465)
stddevs = (0.2470, 0.2435, 0.2616)
img_shape = (3, 32, 32)
num_classes = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NormalizeInput(nn.Module):
    def __init__(self, mean, std, channels=3) :
        super(NormalizeInput, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.channels = channels
        
    def forward(self, input):
        mean = self.mean.reshape(1,self.channels, 1, 1)#.to(device)
        std = self.std.reshape(1, self.channels, 1, 1)#.to(device)
        return (input - mean) / std

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # self.normalize = NormalizeInput(mean=means, std=stddevs, channels=img_shape[0])
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.fc1 = nn.Linear(4096, 150)
        self.fc2 = nn.Linear(150, 10)
    
    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        # x = self.normalize(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def remove_normalize(net, model_torch_load):
    dict_val = {}
    for name, param in zip(model_torch_load.keys(),
            model_torch_load.values()):
        if name != 'normalize.mean' and name != 'normalize.std':
            dict_val[name] = param
    net.load_state_dict(dict_val)
    net.eval()
    return(net)





def load_model_weights_bias(weights_path, device=device):
    net = Network().to(device)
    model_torch_load = torch.load(weights_path)
    net = remove_normalize(net, model_torch_load)    
    net.eval()
    net = net.to(device)
    return(net)

