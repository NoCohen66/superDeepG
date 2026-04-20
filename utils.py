import matplotlib.pyplot as plt
import numpy as np
import yaml
import importlib
import torch
import time
import gc
import os, sys
import torch.onnx


    
def load_yaml_file(path):
    """
    Load a YAML configuration file from the given path.
    Raises FileNotFoundError if the file is not found.
    """
    try:
        with open(path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {path}")



def display_images(original_image, rotated_image, title='Rotated Image'):
    try:
        original_image_np = original_image.squeeze().numpy()
    except: 
        original_image_np = original_image.cpu().squeeze().numpy()
    try:
        rotated_image_np = rotated_image.squeeze().numpy()
    except:
        try:
            rotated_image_np = rotated_image.detach().squeeze().numpy()
        except:
            rotated_image_np = rotated_image.cpu().detach().squeeze().numpy()

    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original_image_np, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[1].imshow(rotated_image_np, cmap='gray')
    axs[1].set_title(title)
    axs[1].axis('off')
    plt.show()

def load_yaml_file(path):
    if path == "experiment/specification/mnist_expe1.yaml":
        print("Warning: Using a default path is configuration path.")
    try:
        with open(path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {path}")


def lirpa_format_transformation(pmin: float, pmax: float, eps: float, display: bool = False):
    """
    This function computes midpoints (theta_midpoints) for sub-intervals between pmin and pmax 
    in increments of eps, and also returns half the interval size (middle_eps) and the number of intervals (inside_split).

    Parameters
    ----------
    pmin : float
        The minimum value of the interval.
    pmax : float
        The maximum value of the interval.
    eps : float
        The step size used to create sub-intervals.
    display : bool, optional
        If True, prints out each sub-interval [midpoint - middle_eps, midpoint + middle_eps].

    Returns
    -------
    theta_midpoints : np.ndarray
        An array of midpoints for each sub-interval.
    middle_eps : float
        Half the interval size (eps / 2), so each sub-interval is [midpoint - middle_eps, midpoint + middle_eps].
    inside_split : int
        The total number of sub-intervals created, which can be used to verify pmax = pmin + inside_split * eps.

    Raises
    ------
    ValueError
        If pmin == pmax.
    ValueError
        If eps < 0.
    """
    if pmin >= pmax:
        raise ValueError("pmin should be inferior to pmax.")
    if eps <= 0:
        raise ValueError("eps must be postive.")

    inside_split = int(round((pmax - pmin) / eps))
    
    if not np.isclose(pmin + inside_split * eps, pmax, atol=1e-8):
         raise ValueError("The relationship pmax = pmin + inside_split * eps isn't respected.")

    theta_edges = np.linspace(pmin, pmax, inside_split + 1)
    theta_midpoints = (theta_edges[:-1] + theta_edges[1:]) / 2.0

    middle_eps = eps / 2.0

    if display:
        for midpoint in theta_midpoints:
            print(f"interval: [{midpoint - middle_eps}, {midpoint + middle_eps}]")

    return theta_midpoints, middle_eps, inside_split
    

def check_config_keys(config, required_keys):
    """
    Check that the 'config' dictionary contains all of the 'required_keys'.
    Raise ValueError if any are missing.
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {', '.join(missing_keys)}")
    else:
        print("The keys have been found.")

def dynamic_import(archi_path, class_name):
    """
    Dynamically import a class from a given module.
    Example: dynamic_import('my_package.my_module', 'MyClass')
    """
    module = importlib.import_module(archi_path)
    ModelClass = getattr(module, class_name)
    model_archi = ModelClass()
    return model_archi



def find_eps_from_inside(theta_min: float, theta_max: float, inside_split: int) -> float:
    """
    This function calculates the value of eps (step size) required to split the interval 
    [theta_min, theta_max] into a specified number of sub-intervals (`inside_split`).

    Parameters
    ----------
    theta_min : float
        The minimum value of the interval.
    theta_max : float
        The maximum value of the interval.
    inside_split : int
        The desired number of sub-intervals to split the interval into.

    Returns
    -------
    eps : float
        The step size required to achieve the specified number of sub-intervals.

    Raises
    ------
    ValueError
        If inside_split is less than or equal to zero.
        If theta_min is greater than or equal to theta_max.
    """
    # Validate inputs
    if inside_split <= 0:
        raise ValueError("inside_split must be greater than 0.")
    if theta_min >= theta_max:
        raise ValueError("theta_min must be less than theta_max.")
    
    # Calculate eps
    eps = (theta_max - theta_min) / inside_split
    return eps

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()





def suppress_stderr():
    fd = sys.stderr.fileno()
    # back up real stderr
    backed_up = os.dup(fd)
    # redirect stderr → /dev/null
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, fd)
    os.close(devnull)
    return backed_up

def restore_stderr(backed_up_fd):
    fd = sys.stderr.fileno()
    os.dup2(backed_up_fd, fd)
    os.close(backed_up_fd)

