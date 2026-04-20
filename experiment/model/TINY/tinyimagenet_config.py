import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn as nn


import sys
sys.path.append('../')

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

means = (0.4819, 0.4457, 0.3934)
stddevs = (0.2734, 0.2650, 0.2770)
img_shape = (3, 56, 56)
num_classes = 200

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model with BatchNorm2D removed (from original CGT architecture)
class Network(nn.Module):
    def __init__(self, means, stddevs, img_shape):
        super().__init__()
        width = 64
        linear_size = 512

        # We have 5 Conv2d layers, each followed by ReLU
        self.model = nn.Sequential(
            nn.Conv2d(3,       width,     3, stride=1, padding=1),  # conv0
            nn.ReLU(),                                            # relu0
            nn.Conv2d(width,   width,     3, stride=1, padding=1),  # conv1
            nn.ReLU(),                                            # relu1
            nn.Conv2d(width,   2*width,   3, stride=2, padding=1),  # conv2
            nn.ReLU(),                                            # relu2
            nn.Conv2d(2*width, 2*width,   3, stride=1, padding=1),  # conv3
            nn.ReLU(),                                            # relu3
            nn.Conv2d(2*width, 2*width,   3, stride=2, padding=1),  # conv4
            nn.ReLU(),                                            # relu4
            nn.Flatten(),                                         # flatten
            nn.Linear(25088, linear_size),                        # fc0
            nn.ReLU(),                                            # relu5
            nn.Linear(linear_size, 200)                           # fc1
        )

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        return self.model(x)




def load_model_weights_bias(weights_path, device=device):
    # 2) Load your fused checkpoint
    fused_ckpt = torch.load(weights_path) 

    # 3) Instantiate the BN‑free model
    new_model = Network(means, stddevs, img_shape)
    new_model.eval()

    # 4) Build a new state_dict mapping old fused conv keys to the new model's conv layers
    new_sd = {}
    # conv layer indices in the fused model were at modules 0,3,6,9,12
    # in our new Sequential they're at positions 0,2,4,6,8
    conv_old_idxs = [0, 3, 6, 9, 12]
    conv_new_idxs = [0, 2, 4, 6, 8]

    for old_i, new_i in zip(conv_old_idxs, conv_new_idxs):
        w_key = f'model.{old_i}.weight'
        b_key = f'model.{old_i}.bias'
        new_w_key = f'model.{new_i}.weight'
        new_b_key = f'model.{new_i}.bias'

        new_sd[new_w_key] = fused_ckpt[w_key]
        new_sd[new_b_key] = fused_ckpt[b_key]

    # copy over the linear layers exactly (they didn’t change position)
    for lin_i in [16, 18]:   # these were your original fc layers
        old_w = fused_ckpt[f'model.{lin_i}.weight']
        old_b = fused_ckpt[f'model.{lin_i}.bias']

        # in new Sequential: fc0 is at index 11, fc1 at 13
        new_layer_idx = 11 if lin_i == 16 else 13
        new_sd[f'model.{new_layer_idx}.weight'] = old_w
        new_sd[f'model.{new_layer_idx}.bias']   = old_b

    # 5) (Optional) If you want to preserve any other parameters (e.g. normalize), copy them too:
    # for k, v in fused_ckpt.items():
    #     if k.startswith('normalize.'):
    #         new_sd[k] = v

    # 6) Load into new model
    new_model.load_state_dict(new_sd)
    return new_model