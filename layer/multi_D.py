import torch
from typing import Callable, Tuple, Optional
import itertools
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#time decorator
import time
import functools
def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        torch.cuda.synchronize()
        end = time.time()
        print(f"{func.__name__} executed in {end - start:.6f} seconds")
        return result
    return wrapper

#time
from contextlib import contextmanager
@contextmanager
def time_block(name="Block"):
    start = time.time()
    yield
    torch.cuda.synchronize()
    end = time.time()
    print(f"{name} executed in {end - start:.6f} seconds")




#@timing_decorator
@torch.no_grad()
def compute_sound_bounds(k_lo: torch.Tensor, k_hi: torch.Tensor, A_lower: torch.Tensor, B_lower: torch.Tensor, A_upper: torch.Tensor,B_upper: torch.Tensor, generate_gx: Callable[[torch.Tensor], torch.Tensor], L_tot_low: torch.Tensor, L_tot_up: torch.Tensor, N: int = 4, ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Add correction term to the bias using Lipschitz optmisation (mean value theorem)"""
    # shapes:
    # k_lo, k_hi: (d,)
    # A_lower, A_upper: (d, C, H, W)
    # B_lower, B_upper: (C, H, W)
    # D: (d, C, H, W)

    device = A_lower.device
    dtype  = A_lower.dtype
    d, C, H, W = A_lower.shape

    # Step 1: build the grid of cell centres (N^d, d)
    delta_k = (k_hi - k_lo) / N                        # (d,)
    grids = [k_lo[i] + (0.5 + torch.arange(N, device=device, dtype=dtype)) * delta_k[i]
             for i in range(d)]
    mesh = torch.meshgrid(*grids, indexing="ij")      # d tensors, each shape (N,)*d
    centres = torch.stack([m.reshape(-1) for m in mesh], dim=1)  # (num_cells, d)
    num_cells = centres.shape[0]
    gx_batch = generate_gx(centres)

    M = C * H * W
    A_flat_lower = A_lower.view(d, M)    # (d, M)
    B_flat_lower = B_lower.view(1, M)    # (1, M)
    A_flat_upper = A_upper.view(d, M)    # (d, M)
    B_flat_upper = B_upper.view(1, M)    # (1, M)

    # Step 2: residuals at all centers and maximal value
    # affine: (num_cells, M)
    affine_lower = centres.to(dtype=dtype, device=device) @ A_flat_lower + B_flat_lower  # broadcast B
    affine_lower = affine_lower.view(num_cells, C, H, W)             # (num_cells, C, H, W)
    residuals_lower = affine_lower - gx_batch 
    m_lower = residuals_lower.amax(dim=0)                        # (C, H, W)

    # Step 3: assemble correction term
    # constant_term_lip used also in upper
    constant_term_lip_lower = (L_tot_low * (delta_k.view(d, 1, 1, 1) / 2)).sum(dim=0)  # (C, H, W)
    corr_lower = m_lower + constant_term_lip_lower  # (C, H, W)
    
    # Step 4: apply correction term to the intercepts D: (d, C, H, W), delta_k: (d,), so broadcast
    B_lower_corr = B_lower - corr_lower[None]  # (1,C, H, W)

    # Step 2 upper: residuals at all centers and maximal value
    # affine: (num_cells, M)
    pred_upper = centres.to(dtype=dtype, device=device) @ A_flat_upper + B_flat_upper  # broadcast B
    pred_upper = pred_upper.view(num_cells, C, H, W)             # (num_cells, C, H, W)
    residuals_upper = gx_batch - pred_upper
    m_upper = residuals_upper.amax(dim=0)                        # (C, H, W)
    # Step 3: assemble correction term
    # constant_term_lip used also in upper
    constant_term_lip_upper = (L_tot_up * (delta_k.view(d, 1, 1, 1) / 2)).sum(dim=0)  # (C, H, W)
    corr_upper = m_upper + constant_term_lip_upper  # (C, H, W)
    # Step 4: apply correction term to the intercepts D: (d, C, H, W), delta_k: (d,), so broadcast
    B_upper_corr = B_upper + corr_upper[None]  # (1, C, H, W)
    return A_lower, B_lower_corr, A_upper, B_upper_corr


#@timing_decorator
def compute_unsound_bounds(input_parameters: torch.Tensor,
                              associated_tr_images: torch.Tensor,
                              max_cond: float = 1e6):
    if True:#with time_block("Unsound: Preparing datas"):
        kappas = input_parameters.to(torch.float64)          # (P, d)
        gx     = associated_tr_images.to(torch.float64)      # (P, C, H, W)

        P, d        = kappas.shape
        P2, C, H, W = gx.shape
        assert P == P2, "Incohérence sur P"

        M        = C * H * W                      # nb total de pixels
        gx_flat  = gx.view(P, M)                  # (P, M)

        inf  = torch.tensor(float("inf"),  dtype=torch.float64, device=kappas.device)
        ninf = torch.tensor(float("-inf"), dtype=torch.float64, device=kappas.device)

        best_res_lower = inf.repeat(M)
        best_res_upper = ninf.repeat(M)

        best_A_lower = torch.full((d, M), float("nan"), dtype=torch.float64, device=kappas.device)
        best_B_lower = torch.full((M,   ), float("nan"), dtype=torch.float64, device=kappas.device)
        best_A_upper = torch.full((d, M), float("nan"), dtype=torch.float64, device=kappas.device)
        best_B_upper = torch.full((M,   ), float("nan"), dtype=torch.float64, device=kappas.device)
        
        EPS_REL = 1e-6
        EPS_ABS = 1e-6


    if True:#with time_block("Unsound: Mask dedicated to mnist"):
        scale_vec = gx_flat.abs().max(dim=0).values      # (M,)
        eps_vec   = EPS_REL * scale_vec + EPS_ABS        # (M,)
        const_mask = scale_vec == 0                      # True si pixel = 0 partout
        
        if const_mask.any():
            # Case where all pixels are 0 during transformation : A = 0, B = 0, res = 0
            best_res_lower[const_mask]  = 0.0
            best_res_upper[const_mask]  = 0.0
            best_A_lower[:, const_mask] = 0.0
            best_A_upper[:, const_mask] = 0.0
            best_B_lower[const_mask]    = 0.0
            best_B_upper[const_mask]    = 0.0
    c = 0
    if True:#with time_block("Unsound: Iterations"):
        for S in itertools.combinations(range(P), d + 1):
            c += 1
            if True:#with time_block(f"Unsound Iteration {c}: step C1"):
                idx = list(S)
                K_sub = kappas[idx]              # (d+1, d)
                G_sub = gx_flat[idx]             # (d+1, M)
            if True:#with time_block(f"Unsound Iteration {c}: step C2"):
                # Step C2
                DeltaK = K_sub[:d] - K_sub[d]    # (d, d)
                DeltaG = G_sub[:d] - G_sub[d]    # (d, M)

            if True:#with time_block(f"Unsound Iteration {c}: step C3"):
                # Step C3: Check affine independence
                cond = torch.linalg.cond(DeltaK)
                if cond > max_cond or cond == float("inf"):
                    continue
                if torch.linalg.matrix_rank(DeltaK) < d:
                    continue

            if True:#with time_block(f"Unsound Iteration {c}: step C4 Solve for slopes"):
                # Step C4: Solve for slopes
                A_cand = torch.linalg.solve(DeltaK, DeltaG)      # (d, M)

            if True:#with time_block(f"Unsound Iteration {c}: step C5: Compute intercept"):
                # Step C5: Compute intercept
                dot_K  = (A_cand * K_sub[d][:, None]).sum(0)     # (M,)
                B_cand = G_sub[d] - dot_K                       # (M,)

            if True:#with time_block(f"Unsound Iteration {c}: step C6: Compute residuals"):
                # Step C6: Compute residuals
                pred   = (kappas @ A_cand) + B_cand.unsqueeze(0)  # (P, M)
                R      = gx_flat - pred                           # (P, M)

            if True:#with time_block(f"Unsound Iteration {c}: step Step C8: Compute average residual"):
                # Step C8: Compute average residual
                avg_res = R.mean(dim=0)                           # (M,)

            if True:#with time_block(f"Unsound Iteration {c}: step Step C9 constraints"):
                # Constraint
                neg = (R < -eps_vec).any(dim=0)
                pos = (R >  eps_vec).any(dim=0)

            if True:#with time_block(f"Unsound Iteration {c}: step Step C7 lower: Remove residual that do not respect the constraint"):
                # Step C7 lower: Remove residual that do not respect the constraint
                lower_mask       = ~neg
                update_lower     = lower_mask & (avg_res < best_res_lower)
                if update_lower.any():
                    best_res_lower[update_lower]   = avg_res[update_lower]
                    best_A_lower[:, update_lower]  = A_cand[:, update_lower]
                    best_B_lower[update_lower]     = B_cand[update_lower]

                # Step C7 upper
                upper_mask       = ~pos #neg & ~pos
                update_upper     = upper_mask & (avg_res > best_res_upper)
                if update_upper.any():
                    best_res_upper[update_upper]   = avg_res[update_upper]
                    best_A_upper[:, update_upper]  = A_cand[:, update_upper]
                    best_B_upper[update_upper]     = B_cand[update_upper]

    if True:#with time_block("Unsound: End"):
        # lower
        if torch.isnan(best_A_lower).all(dim=0).any():
            raise RuntimeError("No lower‐bound found for any pixel – relax EPS_REL, increase P")
        if torch.isnan(best_A_upper).all(dim=0).any():
            raise RuntimeError( "No upper‐bound found for any pixel relax EPS_REL, increase P " )


        # Shape (d, C, H, W) / (C, H, W)
        A_lower = best_A_lower.view(d, C, H, W)
        B_lower = best_B_lower.view(1, C, H, W)
        A_upper = best_A_upper.view(d, C, H, W)
        B_upper = best_B_upper.view(1,C, H, W)

    return A_lower, B_lower, A_upper, B_upper

#@timing_decorator
def sample_hyperbox(bounds, k, device    : Optional[torch.device]    = None, dtype     : torch.dtype               = torch.float32) -> torch.Tensor:

    def to_scalar(x) -> float:
        if torch.is_tensor(x):
            return float(x.item())
        if isinstance(x, np.ndarray):
            return float(x.reshape(()))
        return float(x)

    device = torch.device('cpu') if device is None else device
    lows   = torch.tensor([to_scalar(lo) for lo, _ in bounds],
                          dtype=dtype, device=device)     # (d,)
    highs  = torch.tensor([to_scalar(hi) for _, hi in bounds],
                          dtype=dtype, device=device)     # (d,)
    d      = lows.numel()

    generator = torch.Generator(device=device)
    generator.manual_seed(4)
    rand = torch.rand((k, d), generator=generator, device=device, dtype=dtype)
    return rand * (highs - lows) + lows                   # (k, d)
