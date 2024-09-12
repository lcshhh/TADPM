import torch
import torch.nn as nn
from einops import rearrange
from models.PCN import PCN
from timm.models.vision_transformer import PatchEmbed, Block
from utils.builder import MODELS 
from models.transformer import TransformerEncoderLayer, TransformerDecoderLayer
# from util import SharedMLP, LinearMLP

@MODELS.register_module()
class GlobalVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latent_dim = config.latent_dim
        self.z_dim = config.z_dim
        self.encoder = Encoder(config)
        self.reparameterizers = nn.ModuleList([Reparameterizer(self.latent_dim,self.z_dim) for _ in range(32)])
        self.linears = nn.ModuleList([nn.Linear(self.z_dim,self.latent_dim) for _ in range(32)])
        self.decoder = Decoder(config)
        self.local_encoders = nn.ModuleList([PCN(config.PCN) for _ in range(32)])
        if config.args.PCN_checkpoint != '':
            checkpoint = torch.load(config.args.PCN_checkpoint)
            for i in range(32):
                self.local_encoders[i].load_state_dict(checkpoint['base_model'])
        self.predictors = nn.ModuleList([nn.Sequential(
            nn.Linear(self.latent_dim,128),
            nn.LeakyReLU(),
            nn.Linear(128,32),
            nn.LeakyReLU(),
            nn.Linear(32,8),
        ) for _ in range (32)])
    
    def decode(self, z):
        B = z.shape[0]
        out = self.decoder(z)
        out = out.view(B, 512, -1)
        return out


    def forward(self, points, centers, mask=None):
        embeddings = []
        for i in range(32):
            embedding = self.local_encoders[i].encode(points[:,i])
            embeddings.append(embedding)
        embedding = torch.stack(embeddings,dim=1)
        x = self.encoder(embedding,mask)
        rep = [self.reparameterizers[i](x[:,i,:]) for i in range(32)] #(z,mu,logvar)

        h=torch.stack([self.linears[i](rep[i][0]) for i in range(32)],dim=1)
        mu=torch.stack([t[1] for t in rep],dim=1)
        log_var=torch.stack([t[2] for t in rep],dim=1)

        xx = self.decoder(h,embedding,mask)
        # reconstructed = torch.stack([self.local_encoders[i].decode(xx[:,i]) for i in range(32)],dim=1)
        predicted_axis = torch.stack([self.predictors[i](xx[:,i]) for i in range(32)],dim=1)
        return mu, log_var, predicted_axis

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, 32, config.latent_dim))
        self.blocks = nn.ModuleList(
            [TransformerEncoderLayer(dim=config.latent_dim, n_heads=8) for _ in range(6)]
        )      

    def forward(self, embedding, features=None, mask=None):
        '''
        points:[bs,n,pt_num,3]
        features:[bs,n,9]
        '''
        # bs = points.shape[0]
        # n = points.shape[1]
        # device = points.device
        # embeddings = []
        # for i in range(n):
        #     embedding = self.local_encoders[i].encode(points[:,i])
        #     embeddings.append(embedding)
        # embedding = torch.stack(embeddings,dim=1)  #[bs,n,256]
        x = embedding + self.pos_embedding
        # x = torch.cat([x,features],dim=-1) #[bs,n,265]
        for blk in self.blocks:
            x = blk(x,mask)
        return x

    def encode(self,x):
        point_feature = self.MLP1(x)

        # get global feature
        global_feature = self.MLP2(point_feature)
        global_feature = torch.max(global_feature, dim=2)[0]
        return point_feature, global_feature

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, 32, config.latent_dim))
        self.blocks =  nn.ModuleList(
            [nn.TransformerDecoderLayer(d_model=config.latent_dim, nhead=8) for _ in range(6)]
        )

    def forward(self, h, embedding, mask=None):
        x = embedding + self.pos_embedding
        for block in self.blocks:
            x=block(x,h,mask)
        return x

class Reparameterizer(nn.Module):

    def __init__(self,input_hidden_size,z_dim):
        super(Reparameterizer, self).__init__()
        self.z_dim=z_dim
        self.linear_mu=nn.Linear(input_hidden_size,z_dim)
        self.linear_sigma=nn.Linear(input_hidden_size,z_dim)

    def forward(self, x):
        '''
        :param z: (..., input_hidden_size)
        :return: (..., z_dim)
        '''
        mu = self.linear_mu(x)
        log_var = self.linear_sigma(x)
        eps = torch.randn_like(torch.exp(log_var)).to(mu.device)
        z = mu + torch.exp(0.5*log_var)*eps
        return z, mu, log_var
