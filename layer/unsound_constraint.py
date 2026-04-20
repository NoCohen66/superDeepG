import torch
import numpy as np
from layer.geometric.InverseTransformationBilinear import InverseTransformationBilinear
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class UnsoundConstraint:
    """
    Compute unsound linear approximations (w*theta + b) that bound the samples pixel values 
    """
    def __init__(self, unsound_samples=None, unsound_samples_batch=None, theta_lower=None, theta_upper=None, image=None, transformation=None,  num_iterations=2000, display_info=False):
        if isinstance(theta_lower, torch.Tensor) and theta_lower.shape == torch.Size([1, 1]):
            theta_lower = theta_lower.item()
            theta_upper = theta_upper.item()
        
        if unsound_samples is not None and unsound_samples < 2:
            raise ValueError("N must be greater than 2.")

        self.unsound_samples = unsound_samples
        self.unsound_samples_batch = unsound_samples_batch
        self.theta_lower = theta_lower
        self.theta_upper = theta_upper
        self.image = image
        self.transformation = transformation
        self.display_info = display_info

        # Used for candidates
        self.inf_val = float('inf')
        self.minus_inf_val = float('-inf')        
        self.inf_t   = torch.tensor(self.inf_val,       device=device)
        self.ninf_t  = torch.tensor(self.minus_inf_val, device=device)
    

    def empirical_step(self):
        """
        Sample transformed versions of the image between the angles [theta_lower, theta_upper].
        self Inputs: 
        --------
        self.theta_upper, self.theta_lower : torch.Tensor 
            Shape [B, 1], containing angles.
        self.image : torch.Tensor 
            Shape [C, H, W], containing image.
        self.unsound_samples : int
            Number of samples 
        self.transformation : string
            Key for the transformation (ex: rotate, scale, ...)

        Returns:
        --------
        theta_sampling : torch.Tensor 
            Shape [B, N, 1], containing sampled angles.
        g_x : torch.Tensor 
            Shape [B, N, C, H, W], containing transformed versions of the image.
        """
        try:
            theta_sampling = np.linspace(self.theta_lower, self.theta_upper, self.unsound_samples)
        except:
            theta_sampling = [self.theta_upper]      

        sampling = torch.linspace(0, 1, self.unsound_samples)[None].to(device)  # shape : [1, N]
        if not isinstance(self.theta_lower, torch.Tensor):
            theta_lower = torch.tensor([self.theta_lower]).to(device)
            theta_upper = torch.tensor([self.theta_upper]).to(device)
        else:
            theta_lower = self.theta_lower
            theta_upper = self.theta_upper            
        theta_lower = theta_lower.view(-1, 1) # shape:  B, 1
        theta_upper = theta_upper.view(-1, 1) # shape:  B, 1
        theta_sampling = theta_lower*sampling+(1-sampling)*theta_upper # B, N
        try: 
            g_x =  InverseTransformationBilinear(self.image, self.transformation).forward(theta_sampling.view(-1,1)).contiguous().float().to(device) # batch*N,C,H,W
            _, C, H, W = g_x.shape
            g_x = g_x.view(-1, self.unsound_samples, C, H, W ) # B, N, C, H, W
            
        except:     
            batch_size = 500
            inverse_op = InverseTransformationBilinear(self.image).to(device)
            batched_outputs = [
                inverse_op.forward(torch.reshape(theta_sampling[:, start_idx : start_idx + batch_size], (-1, 1)))
                for start_idx in range(0, self.unsound_samples, batch_size)
            ] #ceach element batched_outputs[0] = B*batch_size, C, H, W
            _, C, H, W = batched_outputs[0].shape
            batched_outputs = [batched_outputs_i.view(-1, batch_size, C, H, W) for batched_outputs_i in batched_outputs]
            g_x = torch.cat(batched_outputs, dim=1).to(device) # [B, N, C, H, W]
        theta_sampling = theta_sampling.view(-1, self.unsound_samples, 1).to(device)  # [B, N,1]
        return theta_sampling, g_x

    def all_candidates(self, theta: torch.Tensor, g_x: torch.Tensor, batch_size:int, start_idx:int): 
        """
        Computes all candidates (even unsafe ones) based on samples
        
        Inputs: 
        theta : torch.Tensor 
            Shape [batch, N, 1], containing sampled angles.
        g_x : torch.Tensor 
            Shape [batch, N, C, H, W], containing transformed versions of the image.
        batch_size: int
        start_idx: int (!warning start_idx must be <= N-batch_size)

        Returns:
        --------
        Theta_6d : torch.Tensor 
            Shape [B,batch_size, N,  1, 1, 1], containing row-col thetas differences (0 on diagonal).
        A_matrix : torch.Tensor 
            Shape [B, batch_size, N,  C, H, W], containing row-col pixels differences (nan on diag).
        """
        theta_flat = theta[..., 0]  # [B, N]
        Theta = theta_flat[:, None, :]  - theta_flat[:, start_idx : start_idx + batch_size, None] # [B, , batch_sizeN]
        # We make sure that the values on the diagonal are at infinity, so that these values are not chosen when calculating the min. When looking for the max, replace +inf with -inf
        # That's why we let diagonal x/0

        Theta_6d = Theta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # => [B,batch_size,N, 1, 1, 1]
        #G = g_x[:, :, None, :, :, :] - g_x[:, None, :, :, :]
        G = g_x[:, None, :, :, :] -  g_x[:, start_idx : start_idx + batch_size, None, :, :, :]  # G => [B,batch_size, N,  C, H, W]
        A_matrix = G / Theta_6d # A_matrix => [B, batch_size,N,  C, H, W]
        return Theta_6d, A_matrix


    def potential_candidates(self,  Theta_6d: torch.Tensor, A_matrix: torch.Tensor, arg_bound: str, display=False):
        """
        Aggregate sample-based slopes into candidate bound coefficients.

        Inputs:
        --------
        Theta_diff : torch.Tensor
            Shape [B, batch_size, N, 1, 1, 1],
            where Theta_diff[b,i,j,0,0,0] = theta_j - theta_{start_idx+i}.
        A_matrix : torch.Tensor
            Shape [B, batch_size, N, C, H, W],
            where A_matrix[b,i,j] = (g_x[b,j] - g_x[b, start_idx+i]) /
                                    (theta_j - theta_{start_idx+i}).
        arg_bound : str, either 'lower' or 'upper'

        Returns:
        --------
        U : torch.Tensor
            Shape [B, batch_size, C, H, W].
            If arg_bound=='lower': U = min over A_matrix where Theta_diff>0.
            If arg_bound=='upper': U = min over A_matrix where Theta_diff<0.
        L : torch.Tensor
            Shape [B, batch_size, C, H, W].
            If arg_bound=='lower': L = max over A_matrix where Theta_diff<0.
            If arg_bound=='upper': L = max over A_matrix where Theta_diff>0.
        """
     
        if arg_bound == 'lower':
            A_Kplus  = torch.where(Theta_6d > 0, A_matrix, self.inf_t)
            A_Kminus = torch.where(Theta_6d < 0, A_matrix, self.ninf_t)
            A_Kminus = torch.where(torch.isinf(A_Kminus), self.ninf_t, A_Kminus)
            L = A_Kminus.max(dim=2).values  # => [B, N, C, H, W]
            U = A_Kplus.min(dim=2).values  # => [B, N, C, H, W]
        else:  # 'upper'
            A_Kplus  = torch.where(Theta_6d > 0, A_matrix, self.ninf_t)
            A_Kminus = torch.where(Theta_6d < 0, A_matrix, self.inf_t)
            A_Kplus = torch.where(torch.isinf(A_Kplus), self.ninf_t, A_Kplus)
            U = A_Kminus.min(dim=2).values  # => [B, N, C, H, W]
            L = A_Kplus.max(dim=2).values   # => [B, N, C, H, W]
        return U, L
            
    def best_candidate(self, U: torch.Tensor, L: torch.Tensor, theta, g_x, expand_theta_mean, g_mean, g_0, compute_cand=False,  display=False):
        """
        Inputs:
        --------
        U : torch.Tensor
            Shape [B, N, C, H, W] potential candidates
        L : torch.Tensor
            Shape [B, N, C, H, W] potential candidates
        expand_theta_mean: torch.Tensor
            Shape [B, 1, 1, 1, 1]
        g_mean: torch.Tensor
            Shape [B, 1, C, H, W]
        g_0: torch.Tensor
            Shape [B, C, H, W]

        Outputs: 
        --------
        A_opt, A_alpha: torch.Tensor
            Shape [B,C,H,W] best slopes candidate
        B_opt: torch.Tensor
            Shape [B,C,H,W] best intercept candidate associated to A_opt, A_alpha

        f(theta;A_opt) = A_opt*theta+B_opt  is an (upper/lower) linear bounds that overestimate (theta,g_x) samples over theta_min, theta_max
        f(theta;A_alpha) = A_alpha*theta+B_opt  is an (upper/lower) linear bounds that overestimate (theta,g_x) samples over theta_min, theta_max
        """
        # Step E
        mask_common = (U >= L)
        maskL = mask_common & (L != self.inf_t) & (L != self.ninf_t)
        maskU = mask_common & (U != self.inf_t) & (U != self.ninf_t)

        # Step F
        # => [B, N, 1,1,1] broadcasting
        theta_4d = theta.unsqueeze(-1).unsqueeze(-1)
        B_L_all = g_x - L * theta_4d  # => [B, N, C, H, W]
        B_U_all = g_x - U * theta_4d # => [B, N, C, H, W]

        # Step G: here we used absolute cost
        cost_L = torch.abs(g_mean - (L * expand_theta_mean + B_L_all))  # => [B,N,C,H,W]
        cost_U = torch.abs(g_mean - (U * expand_theta_mean + B_U_all))  # => [B,N,C,H,W]
       
        fill_invalid = self.inf_t
        cost_L = cost_L.masked_fill(~maskL, fill_invalid)  # => [B,N,C,H,W]
        cost_U = cost_U.masked_fill(~maskU, fill_invalid)  # => [B,N,C,H,W]

        p_L = cost_L.argmin(dim=1, keepdim=True)  # => [B,1,C,H,W]
        p_U = cost_U.argmin(dim=1, keepdim=True)  # => [B,1,C,H,W]

        val_L = cost_L.gather(dim=1, index=p_L)   # => [B,1,C,H,W]
        val_U = cost_U.gather(dim=1, index=p_U)   # => [B,1,C,H,W]

        A_L = L.gather(dim=1, index=p_L)          # => [B,1,C,H,W]
        B_L = B_L_all.gather(dim=1, index=p_L)    # => [B,1,C,H,W]
        A_U = U.gather(dim=1, index=p_U)          # => [B,1,C,H,W]
        B_U = B_U_all.gather(dim=1, index=p_U)    # => [B,1,C,H,W]

        pick_L = (val_L < val_U)                  # => [B,1,C,H,W]

        A_opt = torch.where(pick_L, A_L, A_U)  # => [B,1,C,H,W]
        B_opt = torch.where(pick_L, B_L, B_U)  # => [B,1,C,H,W]

        if display:
            args_to_display = {
                "theta": theta,         # [B, N, 1]
                "g_x": g_x,             # [B, N, C, H, W]
                "L": L,                      # [B, N, C, H, W]
                "U": U,                      # [B, N, C, H, W]
                "B_L_all": (g_x - L * theta_4d),  # all B_L, before gather
                "B_U_all": (g_x - U * theta_4d),  # all B_U, before gather
                "maskL": maskL,              # [B, N, C, H, W]
                "maskU": maskU,              # [B, N, C, H, W]
                "A_opt_final": A_opt.squeeze(1),        # [B, C, H, W]
                "B_opt_final": B_opt.squeeze(1),        # [B, C, H, W]
            }
            return args_to_display
        else:
            return A_opt.squeeze(1), B_opt.squeeze(1), None

    def find_bounds(self, theta: torch.Tensor, g_x: torch.Tensor, compute_cand=False, display=False):
        # theta [B, N, 1]
        # g_x  [B, N, C, H, W]
        g_0 = None
        theta_mean = theta.mean(dim=1, keepdim=True)       # [B, 1, 1]
        g_mean = g_x.mean(dim=1, keepdim=True)        # [B, 1, C, H, W]
        expand_theta_mean = theta_mean.unsqueeze(-1).unsqueeze(-1)   # => [B, 1, 1, 1, 1]

        U_lowers, L_lowers = [], []
        U_uppers, L_uppers = [], [] 
        
       
        for start_idx in range(0, self.unsound_samples, self.unsound_samples_batch): 
            Theta_6d, A_matrix = self.all_candidates(theta, g_x, self.unsound_samples_batch, start_idx)            
            U_lower, L_lower = self.potential_candidates(Theta_6d, A_matrix, "lower") 
            U_lowers.append(U_lower) 
            L_lowers.append(L_lower) 
            U_upper, L_upper = self.potential_candidates(Theta_6d, A_matrix,  "upper") 
            U_uppers.append(U_upper) 
            L_uppers.append(L_upper) 
            del Theta_6d, A_matrix

        U_lower_batch = torch.cat(U_lowers, dim=1)
        L_lower_batch = torch.cat(L_lowers, dim=1)
        U_upper_batch = torch.cat(U_uppers, dim=1)
        L_upper_batch = torch.cat(L_uppers, dim=1)
        

        if display == False: 
            A_lower, B_lower, A_cand_lower = self.best_candidate(U_lower_batch, L_lower_batch, theta, g_x, expand_theta_mean, g_mean, g_0, compute_cand)
            A_upper, B_upper, A_cand_upper = self.best_candidate(U_upper_batch, L_upper_batch, theta, g_x, expand_theta_mean, g_mean, g_0, compute_cand)
            return A_lower, B_lower, A_cand_lower, A_upper, B_upper, A_cand_upper
        else:
            data_lower = self.best_candidate(U_lower_batch, L_lower_batch, theta, g_x, expand_theta_mean, g_mean, g_0, compute_cand, display=True)
            data_upper = self.best_candidate(U_upper_batch, L_upper_batch, theta, g_x, expand_theta_mean, g_mean, g_0, compute_cand, display=True)
            return data_lower, data_upper

