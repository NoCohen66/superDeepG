
import torch
from auto_LiRPA.operators import BoundOptimizableActivation
from auto_LiRPA.operators.clampmult import multiply_by_A_signs

from layer.unsound_constraint import UnsoundConstraint
from layer.sound_constraint import SoundConstraint
from layer.multi_D import compute_unsound_bounds, compute_sound_bounds, sample_hyperbox
from layer.geometric.InverseTransformationBilinear import InverseTransformationBilinear  
# I have add from .clampmult import * in /opt/anaconda3/envs/python311/lib/python3.11/site-packages/auto_LiRPA/operators __init__

# Silence all tracer warnings
import warnings
from torch.jit import TracerWarning
import functools
import logging
warnings.filterwarnings("ignore", category=TracerWarning)
logging.getLogger('onnx').setLevel(logging.ERROR)
logging.getLogger('torch.onnx').setLevel(logging.ERROR)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#class TransformationInterpolationOp(torch.autograd.Function):
    #@staticmethod
    #def symbolic(g, x, const, KKT_samples, KKT_samples_batch, L, N, N_batch, transformation, theta_eps, method, alpha_tr ):
    #    return g.op('custom::TransformationInterpolation', x, const, KKT_samples, KKT_samples_batch, L, N, N_batch, transformation, theta_eps, method, alpha_tr)
class TransformationInterpolationOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, const, KKT_samples, KKT_samples_batch, L, N, N_batch, transformation, theta_eps, method, alpha_tr ):
        node =  g.op('custom::TransformationInterpolation', x, const, KKT_samples, KKT_samples_batch, L, N, N_batch, transformation, theta_eps, method, alpha_tr)
        return node.setType(const.type())

    @staticmethod
    def forward(ctx, x, const, KKT_samples, KKT_samples_batch, L, N, N_batch, transformation, theta_eps, method, alpha_tr):
        
        device = x.device
        res = InverseTransformationBilinear(const.squeeze(0), transformation).forward(x).contiguous().float().to(device)
        return  res


class TransformationInterpolation(torch.nn.Module):
    """Simple module for a custom autograd Transformation."""
    def __init__(self, const, KKT_samples, KKT_samples_batch, L, N, N_batch, transformation, theta_eps, method, alpha_tr):
        super().__init__()
        self.const = const
        self.KKT_samples = KKT_samples
        self.KKT_samples_batch = KKT_samples_batch
        self.L = L
        self.N = N
        self.N_batch = N_batch
        self.transformation = transformation
        self.theta_eps = theta_eps
        self.method = method
        self.alpha_tr = alpha_tr

    def forward(self, x):
        
        transformed_image = TransformationInterpolationOp.apply(x, self.const, self.KKT_samples, self.KKT_samples_batch, self.L, self.N, self.N_batch, self.transformation, self.theta_eps, self.method, self.alpha_tr) #batch, C, H, W
        return transformed_image
class BoundTransformationInterpolation(BoundOptimizableActivation): 
    """
    Custom Bound class handling backward bound propagation for our Transformation operation. 
    """
    #@timing_decorator
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.options = options
        self.ibp_intermediate = True 
        self.splittable = True  
        self.split_beta_used = True 
        self.history_beta_used = False 
        self.flattened_nodes = None
        self.patch_size = {} 
        self.cut_used = False 
        self.cut_module = None
        self.alpha_size = 2

        self.inputs = [inputs[0]]
        self.const = inputs[1].value
        self.KKT_samples = int(inputs[2].value.item())
        self.KKT_samples_batch = int(inputs[3].value.item())
        self.L = inputs[4].value # tensor shape 1, C, H, W
        self.N = int(inputs[5].value.item())
        self.N_batch = int(inputs[6].value.item())
        transformation_tensor = inputs[7].value.item()
        self.theta_eps = inputs[8].value.item()
        method = inputs[9].value.item()
        alpha_tr = inputs[10].value.item()


        # TRANSFORMATION POSSIBILITIES 
        if transformation_tensor == torch.tensor([0]):
            self.transformation = "rotate"
        elif transformation_tensor == torch.tensor([1]):
            self.transformation = "scale"
        elif transformation_tensor == torch.tensor([2]):
            self.transformation = "shear"
        elif transformation_tensor == torch.tensor([3]):
            self.transformation = "translate"
        else:
            raise NotImplementedError(f"The transformation is not implemented yet. Only 'rotate' and 'scale' are available. You set={transformation}")

        # METHOD POSSIBILITIES
        if method == torch.tensor([0]):
            self.method = "CROWN"
        elif method == torch.tensor([1]):
            self.method = "alpha-CROWN"
        elif method == torch.tensor([2]):
            self.method = "CROWN-IBP"
        else:
            raise NotImplementedError(f"The method is not implemented yet. Only 'CROWN', 'alpha-CROWN' and 'IBP' are available. You set={method}")
        
        # ALPHA_TR POSSIBILIES
        if alpha_tr == torch.tensor([0]): 
            self.alpha_tr = False
        elif alpha_tr == torch.tensor([1]):
            self.alpha_tr = True
        else: 
            raise ValueError(f"alpha_tr argument must be 0 or 1. You set={alpha_tr}")
        
        self.mode = options.get("conv_mode", "matrix")

        if self.transformation != "translate":
        
            unsound_constraint = UnsoundConstraint(
                unsound_samples=self.KKT_samples, 
                unsound_samples_batch=self.KKT_samples_batch, 
                theta_lower=inputs[0].value-self.theta_eps, 
                theta_upper=inputs[0].value+self.theta_eps, 
                image=self.const.squeeze(0), 
                transformation=self.transformation)
            theta_values, pixels_values = unsound_constraint.empirical_step()  #shape [batch, N, 1], [batch, N, C, H, W]
            sound = SoundConstraint(
                theta_min_global=inputs[0].value-self.theta_eps,
                theta_max_global=inputs[0].value+self.theta_eps,
                L=self.L,
                partitions=self.N, 
                partitions_batch=self.N_batch, 
                transformation=self.transformation,
                image=self.const.squeeze(0))
    
            
            
            with torch.no_grad():
             
                w_low, b_low, _, w_up, b_up, _ = unsound_constraint.find_bounds(theta_values, pixels_values, compute_cand=False) # expected w_low.shape = torch.Size([B, C, W, H]) (same for others)
                w_low, b_low, w_up, b_up = w_low.to(device), b_low.to(device), w_up.to(device), b_up.to(device)
            
                centers, inverse_transformation_bilinear_theta = sound.precompute()            
                centers = centers # expected centers.shape = torch.Size([self.N-1])
                # lipchitz_term [B, p-1,  1, 1, 1]     
             
                L_tot_low =   w_low.unsqueeze(1).abs() + self.L[None].abs()
                L_tot_up =    self.L[None].abs() + w_up.unsqueeze(1).abs()     
                #inverse_transformation_bilinear_theta = inverse_transformation_bilinear_theta # expected shape = torch.Size([self.N-1, C, W, H])
                
                updated_b_low = sound.sound_constraint(bound="lower",
                            centers=centers, 
                            w_prime=w_low, # [B, C, W, H]
                            b_prime=b_low, #[B, C, W, H] 
                            lipchitz_term=L_tot_low, # [B,p-1,  1, 1, 1]
                            inverse_transformation_bilinear_theta = inverse_transformation_bilinear_theta) #[B,p-1,  C, W, H]
                
                
                new_b_low = b_low - updated_b_low # expected updated_b_low.shape = torch.Size([B, C, W, H])
                
                updated_b_up = sound.sound_constraint(bound="upper",
                                    centers=centers, 
                                    w_prime=w_up,
                                    b_prime=b_up, 
                                    lipchitz_term=L_tot_up,
                                    inverse_transformation_bilinear_theta = inverse_transformation_bilinear_theta) # expected updated_b_up.shape = torch.Size([B,C, W, H])
            
                new_b_up = b_up + updated_b_up
                self.w_low = w_low.unsqueeze(1) # torch.Size([batch, 1, C, W, H])
                self.w_up = w_up.unsqueeze(1) # torch.Size([batch, 1, C, W, H])
                self.new_b_low = new_b_low.unsqueeze(1) # torch.Size([batch, 1, C, W, H])
                self.new_b_up = new_b_up.unsqueeze(1) # torch.Size([batch, 1, C, W, H])
           
                
        else: 
            # -------- inputs info ------------------
            inputs = inputs[0].value #torch.Size([batch,d])
            batch_param = inputs.shape[0] #batch
            d = inputs.shape[1] #batch
            C, H, W = self.const.squeeze(0).shape
            P = int(self.KKT_samples)
            transformer = InverseTransformationBilinear(self.const.squeeze(0), transformation="translate")
         
            # -------- unsound ------------------
            lower_A_list = []
            upper_A_list = []
            lower_bias_list = []
            upper_bias_list = []

            
            for batch_i in range(batch_param):
                k_lo = inputs[batch_i, :]-self.theta_eps #torch.Size([d])
                k_hi = inputs[batch_i, :]+self.theta_eps #torch.Size([d])
                translate_x_min, translate_y_min = k_lo
                translate_x_max, translate_y_max = k_hi
                max_attempts = 8
                for attempt in range(max_attempts):
                    try:
                        pts = sample_hyperbox( [(translate_x_min, translate_x_max),(translate_y_min, translate_y_max)], P + (attempt)*4, device=device)
                        params = torch.tensor(pts, dtype=torch.float32, device=self.const.device)
                        transformed_images = transformer.forward(params)  # (P, C, H, W)
                        lower_A, lower_bias, upper_A, upper_bias =  compute_unsound_bounds(params, transformed_images) # d, C, H, W for w and # 1, C, H, W for b
                        break
                    except Exception as e:
                        print("-" * 100)
                        print(f"{attempt + 1}/{max_attempts} failed : {e}")
                        if attempt == max_attempts - 1:
                            print( f"{max_attempts} tries. "  )
                            lower_A = torch.zeros(d, C, H, W).to(device)
                            upper_A = torch.zeros(d,C, H, W).to(device)
                            lower_bias = torch.zeros(1,C, H, W).to(device)
                            upper_bias = torch.zeros(1,C, H, W).to(device)
                # lower_A torch.Size([d, C, H, W])
                # upper_A torch.Size([d, C, H, W])
                # lower_bias torch.Size([1, C, H, W])
                # upper_bias torch.Size([1, C, H, W])
                L_tot_low =   lower_A.abs() + self.L.abs()
                L_tot_up =    self.L.abs() + upper_A.abs() 
            
                # -------- sound ------------------
                lower_A, lower_bias, upper_A, upper_bias = compute_sound_bounds(k_lo, k_hi,lower_A, lower_bias,upper_A, upper_bias,transformer.forward,L_tot_low,L_tot_up,N=int(self.N))
                # sound output expected:
                # lower_A torch.Size([d, C, H, W])
                # upper_A torch.Size([d, C, H, W])
                # lower_bias torch.Size([1, C, H, W])
                # upper_bias torch.Size([1, C, H, W])
                
                lower_A_list.append(lower_A)
                upper_A_list.append(upper_A)
                lower_bias_list.append(lower_bias)
                upper_bias_list.append(upper_bias)
                
            self.w_low = torch.stack(lower_A_list, dim=0) # batch, d, C, H, W
            self.w_up = torch.stack(upper_A_list, dim=0) # batch, d, C, H, W
            self.new_b_low = torch.stack(lower_bias_list, dim=0) # batch, 1, C, H, W
            self.new_b_up = torch.stack(upper_bias_list, dim=0) # batch, 1, C, H, W

            
            

    def forward(self, x, *args, **kwargs):
        # rotation angle in degrees, negative sign
        device = x.device
        tr_image = InverseTransformationBilinear(self.const.squeeze(0), self.transformation).forward(x).contiguous().float().to(device)
        return  tr_image #torch.Size([batch, C, H, W])


    def _to_dtype(self, t, dtype):
        if t is None or t.dtype == dtype:
            return t
        return t.to(dtype)


    def bound_backward(self,  last_lA, last_uA,  x, start_node=None, unstable_idx=None, *args, **kwargs):
        """
        Bounds propagation (backward)

        Shapes:
        #last_lA.shape torch.Size([CROWN, batch, C, H, W])
        
        #self.w_low [d, batch, C, H, W]
        #self.w_up [d, batch, C, H, W]
        #self.new_b_low [1, batch, C, H, W]
        #self.new_b_up [1, batch, C, H, W]
        """
        last_lA = last_lA.unsqueeze(2) #torch.Size([CROWN,batch, 1, C, H, W])
        last_uA = last_uA.unsqueeze(2) #torch.Size([CROWN,batch, 1, C, H, W])

        if last_lA is not None:
            ref_dtype = last_lA.dtype
        elif last_uA is not None:
            ref_dtype = last_uA.dtype
        else:                                 
            ref_dtype = self.w_low.dtype      

        lA = self._to_dtype(self.w_low, ref_dtype)
        uA = self._to_dtype(self.w_up, ref_dtype)
        lbias = self._to_dtype(self.new_b_low, ref_dtype)
        ubias = self._to_dtype(self.new_b_up, ref_dtype)

        uA1     = uA[None] #torch.Size([1, batch, d, C, H, W])
        lA1     = lA[None]  #torch.Size([1, batch, d, batch, C, H, W])
        ubias1  = ubias[None]  #torch.Size([1, batch, 1, C, H, W])
        lbias1  = lbias[None]  #torch.Size([1, batch, 1, C, H, W])
        
        uA_full, ubias_li = multiply_by_A_signs(last_uA, uA1, lA1, ubias1, lbias1)  #torch.Size([CROWN,d, batch, C, H, W]), torch.Size([CROWN, 1])
        lA_full, lbias_li = multiply_by_A_signs(last_lA, lA1, uA1, lbias1, ubias1)  #torch.Size([CROWN,d, batch, C, H, W]), torch.Size([CROWN, 1])
     
        new_uA_li = uA_full.sum(dim=( 3, 4,5), keepdim=False) #torch.Size([CROWN,batch, d]) 
        new_lA_li = lA_full.sum(dim=( 3, 4,5), keepdim=False) #torch.Size([CROWN,batch, d]) 

        return [(new_lA_li, new_uA_li)], lbias_li, ubias_li
  

    def interval_propagate(self, *v):
        """
        Interval Bound Propagation (IBP) - forward.
        Shapes:
        #self.w_low [d, batch, C, H, W]
        #self.w_up [d, batch, C, H, W]
        #self.new_b_low [1, batch, C, H, W]
        #self.new_b_up [1, batch, C, H, W]
        #param_lower [batch, d] (same param_upper)
        """
        # -----------LOWER------------
        param_lower, param_upper = v[0]
        param_lower = param_lower.unsqueeze(2).unsqueeze(3).unsqueeze(4) #[batch, d, 1,1,1]
        param_upper = param_upper.unsqueeze(2).unsqueeze(3).unsqueeze(4) #[batch, d, 1,1,1]
        
        # -----------LOWER------------
        low_low = self.w_low * param_lower #[batch, d, C, H, W]
        low_up =  self.w_low * param_upper #[batch, d, C, H, W]
        ref_dtype = param_lower.dtype      
        min_contrib = torch.min(low_low, low_up) #[batch, d, C, H, W]
        # sum as matrix mul
        lower_bound = min_contrib.sum(dim=1, keepdim=False) + self.new_b_low.squeeze(1) #[batch, C, H, W]
        
        
        # -----------UPPER------------
        up_low =  self.w_up * param_lower
        up_up =  self.w_up * param_upper
        max_contrib = torch.max(up_low, up_up)
        upper_bound = max_contrib.sum(dim=1, keepdim=False) + self.new_b_up.squeeze(1) #[batch, C, H, W]

        # -----------TYPE------------
        upper_bound = self._to_dtype(upper_bound, ref_dtype) #[batch, C, H, W]
        lower_bound = self._to_dtype(lower_bound, ref_dtype) #[batch, C, H, W]
        return lower_bound, upper_bound
