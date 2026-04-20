import torch
from layer.geometric.InverseTransformationBilinear import InverseTransformationBilinear  
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SoundConstraint:
    def __init__(self, theta_min_global=None, theta_max_global=None, L=None, partitions=None, partitions_batch=None, transformation=None, image=None):
        self.theta_min_global = theta_min_global
        self.theta_max_global = theta_max_global
        self.L = L
        self.partitions = partitions
        self.partitions_batch = partitions_batch
        self.transformation = transformation
        self.image = image # C, W, H

    def inverse_transformation_bilinear(self, theta_tensor): # theta shape batch, self.partitions
        inverse_rot_op = InverseTransformationBilinear(self.image, self.transformation).to(device)
        B = theta_tensor.shape[0]
        batch_size = self.partitions_batch
        batched_outputs = [
            inverse_rot_op.forward(torch.reshape(theta_tensor[:, start_idx : start_idx + batch_size],(-1, 1))).to(device)
            for start_idx in range(0, theta_tensor.shape[1], batch_size)]
        _, C, H, W = batched_outputs[0].shape
        batched_outputs = [batched_i.view(B, -1, C, H, W ) for batched_i in batched_outputs]
        final_output = torch.cat(batched_outputs, dim=1)
        return final_output.to(device)

    def center(self, theta_min, theta_max):
        """
        Computes the midpoint values between theta_min and theta_max : Vectorized center calculation
        Args:
            theta_min (torch.Tensor or float): Minimum theta values #torch.Size([self.partitions-1])
            theta_max (torch.Tensor or float): Maximum theta values #torch.Size([self.partitions-1])
        Returns:
            torch.Tensor : Midpoint values #torch.Size([self.partitions-1])
        """
        return (theta_max + theta_min) / 2


    #@timing_decorator
    def precompute(self):
        """
        # outputs:
        # centers : torch.Size([batch, self.partitions])
        # lipchitz_term : torch.Size([batch,self.partitions,1,1,1])
        # inverse_transformation_bilinear_theta : torch.Size([batch, self.partitions, C, H, W])
        """
        sampling = torch.linspace(0, 1, self.partitions).view(1, -1).to(device) #[1,p]
        if not isinstance(self.theta_min_global, torch.Tensor):
            theta_min_global = torch.tensor([self.theta_min_global]).to(device)
            theta_max_global = torch.tensor([self.theta_max_global]).to(device)
        else:
            theta_min_global = self.theta_min_global
            theta_max_global = self.theta_max_global    
        intervals = sampling*theta_max_global.view(-1, 1) + (1-sampling)*theta_min_global.view(-1, 1) # batch, self.partitions
        all_theta_min = intervals[:,:-1] #torch.Size([batch, self.partitions-1])
        all_theta_max = intervals[:, 1:]  #torch.Size([batch, self.partitions-1]))
        centers = self.center(all_theta_min, all_theta_max) #torch.Size([batch, self.partitions-1]))
        # this lipschitz term does not contains anymore the L
        inverse_transformation_bilinear_theta = self.inverse_transformation_bilinear(centers) #torch.Size([batch, self.partitions-1, C, H, W]))
        return centers.to(device),  inverse_transformation_bilinear_theta.to(device)



    def unsound_bound(self, theta, w_prime, b_prime):
        """
        Computes the unsound lower bound (given by LP) wl' * theta + bl'
        Args:
            theta (torch.Tensor): Input tensor #torch.Size([batch,self.partitions-1])
        Returns:
            torch.Tensor: unsound_lower_bound(theta) #torch.Size([batch,self.partitions-1, C, H, W])
            
        #self.bl_prime torch.Size([batch, C, H, W])
        #self.wl_prime torch.Size([batch, C, H, W])
        #output torch.Size([batch, self.partitions-1, C, H, W])
        """
        #return b_prime + w_prime * theta.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return b_prime[:,None] + w_prime[:,None] * theta.unsqueeze(2).unsqueeze(3).unsqueeze(4)


    def f_lower(self, theta=None, w_prime=None, b_prime=None, inverse_transformation_bilinear_theta=None):
        """
        Computes the exact error function at a given point using the given b' and w'
        """
        return self.unsound_bound(theta, w_prime, b_prime) - inverse_transformation_bilinear_theta 


    def f_upper(self, theta=None, w_prime=None, b_prime=None, inverse_transformation_bilinear_theta=None):
        """
        Computes the exact error function at a given point using the given b' and w'
        """
       
        return  inverse_transformation_bilinear_theta - self.unsound_bound(theta, w_prime, b_prime)

    def f_bound(self, bound=None, centers=None, w_prime=None, b_prime=None, lipchitz_term=None, inverse_transformation_bilinear_theta=None):
        if bound == "lower":
            # the difference between theta_max and theta_min is similar within the batch
            term = (self.theta_max_global[0].squeeze(-1)-self.theta_min_global[0].squeeze(-1))/(2*self.partitions)
            lipschitz_term_all = lipchitz_term*term
            correction = self.f_lower(theta=centers, w_prime=w_prime, b_prime=b_prime, inverse_transformation_bilinear_theta=inverse_transformation_bilinear_theta) + lipschitz_term_all
            return correction
        elif bound == "upper":
            # the difference between theta_max and theta_min is similar within the batch
            term = (self.theta_max_global[0].squeeze(-1)-self.theta_min_global[0].squeeze(-1))/(2*self.partitions)
            lipschitz_term_all = lipchitz_term*term
            lipschitz_term_all = lipchitz_term*term
            correction = self.f_upper(theta=centers, w_prime=w_prime, b_prime=b_prime, inverse_transformation_bilinear_theta=inverse_transformation_bilinear_theta) + lipschitz_term_all
            return correction

    def sound_constraint(self, bound=None, centers=None, w_prime=None, b_prime=None, lipchitz_term=None, inverse_transformation_bilinear_theta = None):
        """
        inputs: 
        centers type: (torch.Tensor) with torch.Size([batch, self.partitions])
        w_prime type: (torch.Tensor) with torch.Size([batch, C, W, H])
        b_prime type: (torch.Tensor) with torch.Size([batch, C, W, H])
        lipchitz_term type: (torch.Tensor) with torch.Size([batch, self.partitions-1, 1, 1, 1]))
        inverse_transformation_bilinear torch.Size([batch, self.partitions-1, C, H, W]))
        
        output: 
        max_f_bound # [batch, C, H, W]
        """
        try: 
            f_bounds = self.f_bound(bound=bound, centers=centers, w_prime=w_prime, b_prime=b_prime, lipchitz_term=lipchitz_term, inverse_transformation_bilinear_theta=inverse_transformation_bilinear_theta) #torch.Size([self.partitions-1,28,28])
            f_bounds, _ = torch.max(f_bounds, dim=1, keepdim=False)
            return f_bounds # [batch, C, H, W], bool
        except RuntimeError as e:  
            if "out of memory" in str(e):
                print(e)
                max_f_bound = None
                if self.partitions_batch > 150000:
                    batch_size = 1000
                else:
                    batch_size = self.partitions_batch
                for start_idx in range(0, self.partitions, batch_size): 
                    f_bounds_i = self.f_bound(bound=bound, centers=centers[:, start_idx : start_idx + batch_size], w_prime=w_prime, b_prime=b_prime, lipchitz_term=lipchitz_term[:, :, :, :], inverse_transformation_bilinear_theta=inverse_transformation_bilinear_theta[:, start_idx : start_idx + batch_size, :, :, :]) #torch.Size([self.partitions-1,28,28])
                    f_bounds_i, _ = torch.max(f_bounds_i, dim=1, keepdim=False)
                    if max_f_bound is None: 
                        max_f_bound = f_bounds_i
                    else: 
                        max_f_bound = torch.max(max_f_bound, f_bounds_i)
                
                return max_f_bound # [batch, C, H, W], bool
            else:
                raise

            