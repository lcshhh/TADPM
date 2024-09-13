import torch
import torch.nn as nn
from einops import rearrange
from models.GlobalVAE import GlobalVAE
from models.diffusion import diffusion
from timm.models.vision_transformer import PatchEmbed, Block

class DiffusionVAE(nn.Module):
    def __init__(self, in_dim, args):
        super().__init__()
        self.vae = GlobalPointVAE(in_dim,args)
        self.condictor = GlobalPointVAE(in_dim,args)
        self.z_dim = args.z_dim
        if args.resume != '':
            checkpoint = torch.load(args.resume)
            self.vae.load_state_dict(checkpoint['model_state_dict'])
            self.condictor.load_state_dict(checkpoint['model_state_dict'])
        # self.vae.requires_grad_(False)
        self.dpm = diffusion(in_features=3)
        # if args.dpm_checkpoint != '':
        #     checkpoint = torch.load(args.dpm_checkpoint)
        #     self.dpm.load_state_dict(checkpoint['model_state_dict'])
    
    def decode(self, z):
        B = z.shape[0]
        out = self.vae.decode(z)
        return out


    def forward(self, pointcloud, axis, before_pointcloud, before_axis):
        latents = self.vae.encode(pointcloud,axis)
        conditions = self.condictor.encode(before_pointcloud, before_axis)
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