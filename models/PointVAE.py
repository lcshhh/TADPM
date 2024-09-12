import torch
import torch.nn as nn
from models.util import SharedMLP, LinearMLP
from utils.builder import MODELS 
# from util import SharedMLP, LinearMLP

@MODELS.register_module()
class PointVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.z_dim = config.z_dim
        self.in_dim = config.in_dim
        self.encoder = Encoder(self.in_dim, self.z_dim)
        self.decoder = Decoder(self.z_dim)
    
    def decode(self, z):
        B = z.shape[0]
        out = self.decoder(z)
        out = out.view(B, 512, -1)
        return out

    def encode(self,x):
        z = self.encoder(x.transpose(1,2))
        return z

    def forward(self, x):
        B, N, C = x.shape
        z = self.encoder(x.transpose(1,2))
        out = self.decoder(z)
        out = out.view(B, 512, C)
        return out

class Encoder(nn.Module):
    def __init__(self, in_dim, z_dim):
        super().__init__()
        self.MLP1 = nn.Sequential(
            SharedMLP(in_dim, 64)
        )
        self.MLP2 = nn.Sequential(
            SharedMLP(64, 64),
            SharedMLP(64, 128),
            SharedMLP(128, 256),
            SharedMLP(256, 512),
        )
        self.fc = nn.Sequential(
            LinearMLP(512, z_dim),
            nn.Linear(z_dim, z_dim)
        )
        # self.fc_mu = nn.Sequential(
        #     LinearMLP(512, z_dim),
        #     nn.Linear(z_dim, z_dim)
        # )
        # self.fc_var = nn.Sequential(
        #     LinearMLP(512, z_dim),
        #     nn.Linear(z_dim, z_dim)
        # )

        # self.fc_global = nn.Sequential(
        #     LinearMLP(512, z_dim),
        #     nn.Linear(z_dim, z_dim)
        # )

    def forward(self, x):
        device = x.device

        # get point feature
        point_feature = self.MLP1(x)

        # get global feature
        global_feature = self.MLP2(point_feature)
        global_feature = torch.max(global_feature, dim=2)[0]
        z = self.fc(global_feature)

        return z

    def encode(self,x):
        point_feature = self.MLP1(x)

        # get global feature
        global_feature = self.MLP2(point_feature)
        global_feature = torch.max(global_feature, dim=2)[0]
        return point_feature, global_feature

class Decoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.SharedMLP = nn.Sequential(
            SharedMLP(1+z_dim, 64),
            SharedMLP(64, 32),
            SharedMLP(32, 16),
            SharedMLP(16, 3),
            nn.Conv1d(3, 3, 1)
        )
        self.fc = nn.Sequential(
            LinearMLP(z_dim, 256),
            LinearMLP(256, 512),
            LinearMLP(512, 1024),
            LinearMLP(1024, 512*3),
            nn.Linear(512*3, 512*3)
        )

    def forward(self, x):
        out = self.fc(x)
        return out

if __name__ == "__main__":
    x = torch.randn(100, 3, 512)
    Net = PointVAE(in_dim=3, z_dim=256)
    mu, log_var, z, out = Net(x)

    print(z.shape)
    print(out.shape)
