import torch
import torch.nn as nn
from einops import rearrange
from models.GlobalVAE import GlobalVAE
from models.diffusion import diffusion
from timm.models.vision_transformer import PatchEmbed, Block
from models.PCN import PCN
from utils.builder import MODELS

@MODELS.register_module()
class DiffusionVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vae = GlobalVAE(config)
        # self.vae.requires_grad_(False)
        self.condictors = nn.ModuleList([PCN(config.PCN) for _ in range(32)])
        self.z_dim = config.z_dim
        if config.args.vae_ckpts != '':
            checkpoint = torch.load(config.args.vae_ckpts)
            self.vae.load_state_dict(checkpoint['base_model'])
        self.dpm = diffusion(config.latent_dim)
    
    def decode(self, z):
        B = z.shape[0]
        out = self.vae.decode(z)
        return out


    def forward(self, pointcloud, axis, before_pointcloud):
        latents = self.vae.encode(pointcloud,axis)
        conditions = torch.stack([self.condictors[i].encode(before_pointcloud[:,i]) for i in range(32)],dim=1)
        predicted_latents = self.dpm(latents, conditions)
        
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