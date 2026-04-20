
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchvision.datasets as datasets

def build_subintervals(lb: torch.Tensor, ub: torch.Tensor, interval_size: float) -> torch.Tensor:
    """Return tensor ``(n_sub, 2, d)`` covering *[lb, ub]* with equal width."""
    length = (ub - lb).squeeze(0)
    n_sub = int(math.ceil((length.max() / interval_size).item())) or 1
    subs = []
    for k in range(n_sub):
        low = lb + k * interval_size
        high = torch.minimum(low + interval_size, ub)
        subs.append(torch.stack((low.squeeze(0), high.squeeze(0)), dim=0))
    return torch.stack(subs, dim=0)


def coordinate_grid(H: int, W: int, *, device, dtype=torch.float64) -> torch.Tensor:
    """Cartesian coordinates (x, y) for every pixel; shape ``(2, H, W)``."""
    i = torch.arange(H, dtype=dtype, device=device)
    j = torch.arange(W, dtype=dtype, device=device)
    ii, jj = torch.meshgrid(i, j, indexing="ij")

    ox, oy = (W + 1) / 2, (H + 1) / 2
    x = jj - ox
    y = oy - ii
    return torch.stack((x, y))  # 2 × H × W


def compute_grad_tr_im(
    *,
    C: int,
    H: int,
    W: int,
    transformations: List[str],
    lb: torch.Tensor,
    ub: torch.Tensor,
    interval_size: float,
    device: torch.device,
) -> torch.Tensor:
    # Source: https://github.com/benbatten/PWL-Geometric-Verification
    # Code authors: Ben Batten, Yang Zheng, Alessandro De Palma, Panagiotis Kouvaros and Alessio Lomuscio
    """Return `(n_sub, 2, n_tr, cand, C, H, W)` ready for broadcasting."""
    sub_intervals = build_subintervals(lb, ub, interval_size).to(device)  # n_sub × 2 × d
    coords = coordinate_grid(H, W, device=device)
    grad_fn_key = "_".join(transformations)
    g = gradient_function_names[grad_fn_key](sub_intervals, [coords[0], coords[1]])
    if g.ndim == 5:
        g = g.unsqueeze(2)  
    g = g.unsqueeze(4).expand(-1, -1, -1, -1, C, -1, -1)
    return g


def compute_grad_interp(image: torch.Tensor, *, device) -> torch.Tensor:
    # Source: https://github.com/benbatten/PWL-Geometric-Verification
    # Code authors: Ben Batten, Yang Zheng, Alessandro De Palma, Panagiotis Kouvaros and Alessio Lomuscio
    if image.ndim == 2:
        image = image[:, :, None]
    g = get_interpolation_gradient_grid(image.to(torch.float64))  # W×H×C×2
    g_max = g.abs().view(-1, image.shape[2], 2).max(dim=0).values  # C×2
    # (1,2,1,1,C,1,1)
    g_max = g_max.T.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(-1).unsqueeze(-1)
    return g_max.to(device)


def compute_grad_product_transformation_interpolation(
    tr_grad: torch.Tensor,  # n_sub × 2 × n_tr × cand × C × H × W
    interp_grad: torch.Tensor,  # 1 × 2 × 1 × 1 × C × 1 × 1
) -> torch.Tensor:
    # Source: https://github.com/benbatten/PWL-Geometric-Verification
    # Code authors: Ben Batten, Yang Zheng, Alessandro De Palma, Panagiotis Kouvaros and Alessio Lomuscio
    # Multiply then sum over (x,y) axis = 1
    prod = (tr_grad * interp_grad).sum(dim=1)            # n_sub × n_tr × cand × C × H × W
    grad = prod.abs().max(dim=0).values                  # max over n_sub
    grad = grad.max(dim=1).values                        # max over cand
    return grad  # n_tr × C × H × W



def _max_rotate_grad_parallel_fixed(active_intervals, xy_grid):
    """
    active_intervals : (n_sub, 2, 1)
    xy_grid          : (2, H, W)  – x puis y
    Return          : (n_sub, 2, 4, H, W)
    """
    # Source: https://github.com/benbatten/PWL-Geometric-Verification
    # Code authors: Ben Batten, Yang Zheng, Alessandro De Palma, Panagiotis Kouvaros and Alessio Lomuscio
    #  Source page 12: https://files.sri.inf.ethz.ch/website/papers/neurips19-deepg.pdf 
    # Rotation operator:
    #   R_φ(x, y) = ( cos(φ)*x  - sin(φ)*y,
    #                   sin(φ)*x  + cos(φ)*y )
    #
    # Derivative w.r.t. phi:
    #   dR/dφ (x, y) = ( -sin(φ)*x  - cos(φ)*y,
    #                       cos(φ)*x  - sin(φ)*y )
    #        d/dφ [dR/dφ] = 0
    #     d/dφ [dR/dφ] = -cosφ * x  + sinφ * y = 0 -> sinφ * y = cosφ * x -> tan φ = x/y    -> φ_s1 = arctan2(x, y)
    #      d/dφ [dR/dφ] = -sinφ * x  - cosφ * y = 0 ->  -sinφ * x  = cosφ * y  -> tan φ = -y/x   -> φ_s2 = arctan2(-y, x)
   
    active_intervals = active_intervals[..., 0]
    
    x, y = xy_grid            # (H, W) chacun
    H, W = x.shape
    n_sub = active_intervals.shape[0]
    # (2, H, W)  ->  (n_sub, 2, H, W)
    static_points = torch.stack((
        torch.atan2(x, y),          # = arctan(x / y) mais sûr quand y = 0
        torch.atan2(-y, x)
    ), dim=0).to(active_intervals.device)
   
    static_points = static_points.unsqueeze(0).expand(n_sub, -1, H, W)
    #   (n_sub, 2, H, W)
    bounds = active_intervals[:, :, None, None].expand(n_sub, 2, H, W)
    candidate_params = torch.cat((static_points, bounds), dim=1)  # (n_sub, 4, H, W)
    #  (n_sub, 2, 4, H, W)
    sin_t, cos_t = torch.sin(candidate_params), torch.cos(candidate_params)
    candidate_grads = torch.stack((
        -x[None, None] * sin_t - y[None, None] * cos_t,
         x[None, None] * cos_t - y[None, None] * sin_t,
    ), dim=1)
    inside = (active_intervals[:, :1, None, None] < static_points) & \
             (static_points < active_intervals[:, 1:, None, None])
    candidate_grads[:, :, :2] *= inside.unsqueeze(1)
    return candidate_grads  # (n_sub, 2, 4, H, W)


def _max_scale_grad_parallel_fixed(active_intervals: torch.Tensor, xy_grid):
    # Source: https://github.com/benbatten/PWL-Geometric-Verification
    # Code authors: Ben Batten, Yang Zheng, Alessandro De Palma, Panagiotis Kouvaros and Alessio Lomuscio
    # Formula source page 12: https://files.sri.inf.ethz.ch/website/papers/neurips19-deepg.pdf 
    # Scaling operator:
    #   Sc_{λ1,λ2}(x, y) = ( λ1 * x,  λ2 * y )
    #
    # Jacobian w.r.t. (λ1, λ2):
    #   ∂Sc/∂(λ1,λ2)(x, y) = [ [ x, 0 ],
    #                         [ 0, y ] ]
    # max Jacobian w.r.t. (λ1, λ2):
    #   ∂Sc/∂(λ1,λ2)(x, y) = [ [ |x|, 0 ],
    #                         [ 0, |y| ] ]
    x, y = xy_grid                        # (H, W)
    n_sub = active_intervals.shape[0]
    
    base = torch.stack((x.abs(), y.abs()), dim=0)      # (2, H, W)
    g = base.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    g = g.repeat(n_sub, 1, 1, 1, 1, 1)                # (n_sub, 2, 1, 1, H, W)
    return g

def _max_shear_grad_parallel_fixed(active_intervals: torch.Tensor, xy_grid):
    # Source: https://github.com/benbatten/PWL-Geometric-Verification
    # Code authors: Ben Batten, Yang Zheng, Alessandro De Palma, Panagiotis Kouvaros and Alessio Lomuscios
    # Shear operator:
    #   Sh_m(x, y) = ( x + m * y,  y )
    #
    # Jacobian w.r.t. m:
    #   ∂Sh/∂m (x, y) = ( y, 0 )
    # max Jacobian w.r.t. m:
    #   ∂Sh/∂m (x, y) = ( |y|, 0 )
    _, y = xy_grid

    #H = y.shape[0]
    #scale_y = (H - 1) / 2.0
    #y = y*scale_y
   
    n_sub = active_intervals.shape[0]
    g = torch.stack((y.abs(), torch.zeros_like(y)), dim=0)           # 2,H,W
    g = g.unsqueeze(0).unsqueeze(2).unsqueeze(3)               # (n_sub,2,1,1,H,W)
    g = g.expand(n_sub, -1, -1, -1, -1, -1)
    return g


def _max_translate_grad_parallel_fixed(active_intervals: torch.Tensor, xy_grid):
    # Source: https://github.com/benbatten/PWL-Geometric-Verification
    # Code authors: Ben Batten, Yang Zheng, Alessandro De Palma, Panagiotis Kouvaros and Alessio Lomuscio
    """ Source page 12: https://files.sri.inf.ethz.ch/website/papers/neurips19-deepg.pdf 
    # Translation operator:
    #   T_{v1,v2}(x, y) = ( x + v1,  y + v2 )
    #
    # Jacobian with respect to (v1, v2):
    #   dT/d(v1, v2)(x, y) = [ [1, 0],
    #                         [0, 1] ]
    """
    n_sub = active_intervals.shape[0]
    H, W = xy_grid[0].shape
    ones = torch.ones((H, W), dtype=torch.float64, device=active_intervals.device)
    zeros = torch.zeros_like(ones)
    # (2, n_tr=2, H, W)
    g = torch.stack((
        torch.stack(( ones, zeros), dim=0),   
        torch.stack((zeros,  ones), dim=0),   
    ), dim=1)
    g = g.unsqueeze(0).unsqueeze(3)                      # (n_sub,2,2,1,H,W)
    g = g.expand(n_sub, -1, -1, -1, -1, -1)
    return g


gradient_function_names = {
    'rotate': _max_rotate_grad_parallel_fixed,
     'scale': _max_scale_grad_parallel_fixed,
     'shear': _max_shear_grad_parallel_fixed,
     'translate' : _max_translate_grad_parallel_fixed,
}



def get_interpolation_gradient_grid(image):
    """
    Computes per-cell upper bounds on the gradients of a bilinearly interpolated image.
    
    For each interpolation cell defined by four corner values (p0, p1, p2, p3),
    we compute:
        - the maximum of |∂I/∂k| over k ∈ [0,1], and
        - the maximum of |∂I/∂l| over l ∈ [0,1].
	
    These correspond to:
        ∂I/∂k = l(p0 + p3 − p2 − p1) + (p2 − p0)
        ∂I/∂l = k(p0 + p3 − p2 − p1) + (p1 − p0)
	
    So:
        max|∂I/∂k| = max(|p2 − p0|, |p3 − p1|)
        max|∂I/∂l| = max(|p1 − p0|, |p3 − p2|)
	
    Args:
        image: A tensor of shape (..., H, W, C) or (..., H, W), representing a grayscale or multi-channel image.
    Returns:
        A tensor of shape (..., H-1, W-1, 2) containing per-cell maximum interpolation gradients along the vertical axis (∂/∂i) and horizontal axis (∂/∂j).
    """
    
    p0 = image[..., :-1, :-1, :]   # top-left
    p1 = image[...,:-1, 1:,  :]   # top-right
    p2 = image[..., 1:,  :-1, :]   # bottom-left
    p3 = image[..., 1:,  1:,  :]   # bottom-right

    slope  = p3 + p0 - p2 - p1          # common slope
    x_int  = p2 - p0                    
    y_int  = p1 - p0                   

    max_dx = torch.maximum(torch.abs(x_int), torch.abs(x_int + slope))
    max_dy = torch.maximum(torch.abs(y_int), torch.abs(y_int + slope))

    # in torch, the y-coordinates is first
    return torch.stack((max_dx, max_dy), dim=-1)   # (H-1,W-1,C, 2)


def get_interpolation_gradient_grid_OLD(image):
    """Returns a grid with shape: (self.image.shape, 2) where the last dimension contains the maximum abs(gradients) wtf x (first entry) and y (second entry)"""

    # First we compute the bracket common to each partial (P_{i+1,j+1} + P_{i, j} - P_{i+1, j} - P_{i, j+1})
    interior_bracket = image.clone()[:1, :1, :] # P_{i,j}
    interior_bracket = interior_bracket + image[1:, 1:, :] - image[:-1, 1:, :] - image[1:, :-1, :]

    didx = image.clone()[:-1, :-1, :] * -1 # P_{i,j}
    didx += image[:-1, 1:, :]

    didy = image.clone()[:-1, :-1, :] * -1  # P_{i,j}
    didy += image[1:, :-1, :]

    # For each pixel we want to either add the constant or not (delta x/y = 0/1), with the goal of finding the maximum absolute gradient.
    didx = torch.maximum(abs(didx), abs(didx+interior_bracket))
    didy = torch.maximum(abs(didy), abs(didy+interior_bracket))
    return torch.stack([didx, didy], dim=-1)
