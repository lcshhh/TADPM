import torch
import torch.nn as nn
from einops import rearrange
from models.voxelUNet import VoxelUNet
from models.diffusion import diffusion
from timm.models.vision_transformer import PatchEmbed, Block
from models.PCN import PCN
from utils.builder import MODELS 
# from util import SharedMLP, LinearMLP


@MODELS.register_module()
class DiffusionVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vae = VoxelUNet(config.vae)
        if config.args.vae_checkpoint!='':
            checkpoints = torch.load(config.args.vae_checkpoint)
            self.vae.load_state_dict(checkpoints['base_model'])
        self.resolution = config.vae.resolution
        self.zdim = config.vae.zdim
        self.vae.requires_grad_(False)
        self.dpm = diffusion()
    
    def decode(self, z):
        B = z.shape[0]
        out = self.vae.decode(z)
        return out


    def forward(self, points):
        # self.vae.eval()
        latents = self.vae.encode(points)
        # input_latents = rearrange(latents,'b c w h d -> b (w h d) c')
        predicted_latents = self.dpm(latents)
        # predicted_latents = rearrange(predicted_latents,'b (w h d) c -> b c w h d',w=2*self.resolution,h=2*self.resolution,d=8*self.resolution)
        # rec, masks = self.vae.decode(predicted_latents)
        
        # out = out.view(B, 512, C)
        return latents,  predicted_latents

        # for point-wise decoder
        z_clone = z.clone().detach()
        z = z.view(B, self.z_dim, 1).repeat(1, 1, N)
        features = torch.concat([point_feature, z], dim=1)

        out = self.decoder(features)
        out = out.view(B, C, N)

        return mu, log_var, z_clone, out
    
    def predict(self, before_pointcloud, before_axis):
        conditions = self.condictor.encode(before_pointcloud, before_axis)
        predicted_latents = self.dpm(None,conditions)
        out = self.decode(predicted_latents)
        # out = out.view(B, 512, C)
        return out
    
    def sample(self,sample_num):
        z = torch.randn(sample_num,self.zdim,2*self.resolution,2*self.resolution,8*self.resolution)
        predicted_latents = self.dpm.ddim_sample(z).float()
        points, masks = self.vae.decode(predicted_latents)
        return points, masks
    



