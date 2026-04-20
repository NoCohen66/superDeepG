import torch
from layer.transformation_interpolation import TransformationInterpolation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from experiment.normalize_utils import NormalizeInput


class WrapperNN(torch.nn.Module):
    def __init__(self, model=None, image=None, KKT_samples=None, KKT_samples_batch=None, L=None, N=None, N_batch=None, transformation=None, theta_eps=None, method=None, alpha_tr=None, means=None, stddevs=None, img_shape=None):
        super(WrapperNN, self).__init__()
        self.layer = TransformationInterpolation(image, KKT_samples, KKT_samples_batch, L, N, N_batch, transformation, theta_eps, method, alpha_tr)
        self.model = model
        self.normalize = NormalizeInput(mean=means, std=stddevs, channels=img_shape[0])


    def forward(self, x):
        x = self.layer(x)
        x = self.normalize(x)
        x = self.model(x)
        return(x)



def set_WrapperNN( model=None, 
                   image=None,
                   KKT_samples=torch.Tensor([100]),
                   KKT_samples_batch=torch.Tensor([100]),
                   L=torch.Tensor([5]),
                   N=torch.Tensor([1000]),
                   N_batch=torch.Tensor([1000]), 
                   transformation='transformation', 
                   theta_eps=torch.Tensor([1]),
                   method='method', 
                   alpha_tr='alpha_tr',
                   means=None, 
                   stddevs=None, 
                   img_shape=None, 
                   device=device):

    # TRANSFORMATION POSSIBILITIES
    if transformation == "rotate":
        transformation_tensor = torch.tensor([0])
    elif transformation == "scale":
        transformation_tensor = torch.tensor([1])
    elif transformation == "shear":
        transformation_tensor = torch.tensor([2])
    elif transformation == "translate":
        transformation_tensor = torch.tensor([3])
    else:
        raise NotImplementedError(f"The transformation is not implemented yet. Only 'rotate' and 'scale' are available. You set={transformation}")
    
     # METHOD POSSIBILITIES
    if method == "CROWN":
        method = torch.tensor([0])
    elif method == "alpha-CROWN":
        method = torch.tensor([1])
    elif method == "CROWN-IBP":
        method = torch.tensor([2])
    else:
        raise NotImplementedError(f"The method is not implemented yet. Only 'CROWN', 'alpha-CROWN' and 'IBP' are available. You set={transformation}")
       
    # ALPHA_TR POSSIBILIES
    if alpha_tr == True: 
        alpha_tr = torch.tensor([1])
    elif alpha_tr == False: 
        alpha_tr = torch.tensor([0])
    else: 
        raise ValueError(f"alpha_tr argument must be True or False. You set={alpha_tr}")


    network = WrapperNN(model=model, image=image, KKT_samples=KKT_samples, KKT_samples_batch=KKT_samples_batch, L=L, N=N, N_batch=N_batch, transformation=transformation_tensor, theta_eps=theta_eps, method=method, alpha_tr=alpha_tr, means=means, stddevs=stddevs, img_shape=img_shape)
    return(network.to(device))


