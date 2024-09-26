import torch
import torch.nn as nn
from models.PCN import PCN
from utils.builder import MODELS 

@MODELS.register_module()
class Predictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoders = nn.ModuleList([PCN(config.PCN) for _ in range(32)])
        self.linears = nn.ModuleList([nn.Sequential(
            nn.Linear(config.latent_dim,512),
            nn.GELU(),
            nn.Linear(512,256),
            nn.GELU(),
            nn.Linear(256,5)
        ) for _ in range(32)])
        
    def forward(self, points):
        encodings = [self.encoders[i].encode(points[:,i]) for i in range(32)]
        predicted_axis = torch.stack([self.linears[i](encodings[i]) for i in range(32)],dim=1)
        return predicted_axis

        
