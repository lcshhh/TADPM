import torch
import torch.nn as nn
from einops import rearrange

class SharedMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU()
        )
    
    def forward(self, x):
        out = self.main(x)
        return out

class LinearMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU()
        )

    def forward(self, x):
        out = self.main(x)
        return out

class GlobalLinearMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # self.main = nn.ModuleList([
        #     nn.Linear(in_dim, out_dim),
        #     nn.BatchNorm1d(out_dim),
        #     nn.LeakyReLU()]
        # )
        self.l = nn.Linear(in_dim, out_dim)
        self.b = nn.BatchNorm1d(out_dim)
        self.lr = nn.LeakyReLU()

    def forward(self, x):
        # for l,b,lr in self.main:
        x = self.l(x)
        x = self.b(x.transpose(1,2)).transpose(1,2)
        out = self.lr(x)
        return out

class LinearMLP2D(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU()
        )

    def forward(self, x):
        out = self.main(x)
        return out

if __name__ == "__main__":
    x = torch.randn(100, 2, 199)
    Net = SharedMLP(2, 10)
    out = Net(x)
    print(x.shape)
    print(out.shape)
