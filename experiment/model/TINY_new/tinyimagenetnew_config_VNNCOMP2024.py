#https://github.com/huanzhang12/vnncomp2024_tinyimagenet_benchmark/tree/main
#Huan Zhang
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import torch
import torch.nn as nn
import sys
sys.path.append('../')
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from onnx2pytorch import ConvertModel
import onnx

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def load_model_weights_bias(weights_path, device=device):
    onnx_model = onnx.load(weights_path)
    model = ConvertModel(onnx_model, experimental=True)
    model.eval().to(device)
    # Sanity check
    for _, p in model.named_parameters():
        assert p.device == device
    for _, b in model.named_buffers():
        assert b.device == device
    print(f"INFO: ONNX loaded and converted: {weights_path}")
    return model