import torch
import warnings
from torch.jit import TracerWarning

# silence all tracer warnings
warnings.filterwarnings("ignore", category=TracerWarning)
class InverseTransformationBilinear(torch.nn.Module):
    """
    Bilinear inverse transformation for images of shape (C, H, W).
    
    Inputs:
      - img (torch.Tensor): shape = (C, H, W) with C >= 1
      - theta (torch.Tensor): angles in degrees shape=(N,)

    Output:
      - transformed_img (torch.Tensor): shape = (N, C, H, W)
    """
    def __init__(self, img: torch.Tensor, transformation=None):
        super().__init__()
        self.transformation = transformation
        self.img = img  # shape = (C, H, W)
        self.C, self.H, self.W = img.shape  # shape = C, H, W,


    def forward(self, theta: torch.Tensor):  
        N = theta.shape[0]
        
        y, x = torch.meshgrid(torch.linspace(-1, 1, self.H).to(theta.device), torch.linspace(-1, 1, self.W).to(theta.device), indexing="ij")
        grid = torch.stack((x, y), dim=-1).to(theta.device)  # Shape (H, W, 2)
        
        # Expand grid for each thetza factor and clone to avoid memory overlap
        grid = grid.unsqueeze(0).repeat(N, 1, 1, 1).clone()  # Shape (N, H, W, 2)
        
        if self.transformation == "rotate" or self.transformation == torch.tensor([0]):
          theta_rad = theta.squeeze(-1) # Theta already converted.

          cos_theta = torch.cos(theta_rad)
          sin_theta = torch.sin(theta_rad)
          inv_rotation = torch.zeros((N, 2, 3), device=theta.device, dtype=theta.dtype)
          inv_rotation[:, 0, 0] = cos_theta
          inv_rotation[:, 0, 1] = -sin_theta
          inv_rotation[:, 1, 0] = sin_theta
          inv_rotation[:, 1, 1] = cos_theta
        
          
          # Prepare the image as a batch.
          # self.img is of shape (C, H, W); add a batch dimension and expand to 
          img_batch = self.img.unsqueeze(0).expand(N, -1, -1, -1).to(theta.device).float() #(N, C, H, W)

          grid = torch.nn.functional.affine_grid(inv_rotation, img_batch.size(), align_corners=True)
          grid = grid.to(theta.device).float()

          
          rotated_img = torch.nn.functional.grid_sample(
              img_batch, grid, mode='bilinear', padding_mode='zeros', align_corners=True
          )
          
          return rotated_img

        elif self.transformation == "scale" or self.transformation == torch.tensor([1]):
          lambda_x = theta
          
          # Apply scaling: x' = x / lambda_x, y' = y / lambda_y
          grid[..., 0] /= lambda_x.view(N, 1, 1)
          grid[..., 1] /= lambda_x.view(N, 1, 1)
          
        elif self.transformation == "shear" or self.transformation == torch.tensor([2]):
          m = theta
          
          shear_x = m.view(N, 1, 1) * grid[..., 1]  # Apply shear to x-coordinates
          grid[..., 0] -= shear_x  # Modify x-coordinates

        elif self.transformation == "translate" or self.transformation == torch.tensor([3]):
          dx = theta[:,0]
          dy = theta[:,1]
          
          # Normalize dx, dy to match the grid scale (-1 to 1)
          dx_norm = dx.view(N, 1, 1) * (2.0 / (self.W-1)) # Shape (N, 1, 1)
          dy_norm = dy.view(N, 1, 1) * (2.0 / (self.H-1)) # Shape (N, 1, 1)
        
          # grid : Shape (N, H, W, 2)
          # dx_norm, dy_norm : Shape (N, 1, 1)
          grid[..., 0] -= dx_norm
          grid[..., 1] -= dy_norm
          
        else: 
          raise NotImplementedError("This transformation has not yet been implemented.")
        
        # Expand image to batch format (1, C, H, W) and repeat for batch processing
        image_batch = self.img.unsqueeze(0).expand(N, -1, -1, -1).to(theta.device).float()  # Shape (N, C, H, W)

        # Perform sampling
        transformed_images = torch.nn.functional.grid_sample(image_batch, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return transformed_images # Shape (N, C, H, W)