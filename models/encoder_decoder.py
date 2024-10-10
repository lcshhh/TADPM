
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.PCN import PCN
from einops import rearrange
from models.transformers.transformer import TransformerEncoderLayer,TransformerDecoderLayer

class Encoder(nn.Module):
    def __init__(self, config, args=None):
        super(Encoder, self).__init__()
        # self.local_encoders = nn.ModuleList([PCN(config) for _ in range(32)])
        # if args.PCN_checkpoint != '':
        #     checkpoint = torch.load(args.PCN_checkpoint)
        #     for i in range(32):
        #         self.local_encoders[i].load_state_dict(checkpoint['base_model'])
        self.pos_embedding = nn.Parameter(torch.randn(1, 32, config.latent_dim))
        self.blocks = nn.ModuleList(
            [TransformerEncoderLayer(d_model=config.latent_dim, nhead=config.n_head) for _ in range(config.depth)]
        )      

    def forward(self, features, mask=None):
        '''
        input:
        points: [bs,32,sample_num,3]

        return:
        features: [bs,32,latent_dim]
        '''
        # features = torch.stack([self.local_encoders[i].encode(points[:,i]) for i in range(32)], dim=1)   # [bs,32,latent_dim]
        features = features + self.pos_embedding
        for block in self.blocks:
            # features = block(features,src_key_padding_mask=mask)
            features = block(features)
        features = rearrange(features,'b (w h) c -> b c w h', w=8)
        return features

class Decoder(nn.Module):
    def __init__(self, config, args=None):
        super(Decoder, self).__init__()
        # self.local_decoders = nn.ModuleList([PCN(config) for _ in range(32)])
        # if args.PCN_checkpoint != '':
        #     checkpoint = torch.load(args.PCN_checkpoint)
        #     for i in range(32):
        #         self.local_decoders[i].load_state_dict(checkpoint['base_model'])
        self.pos_embedding = nn.Parameter(torch.randn(1, 32, config.latent_dim))
        self.blocks = nn.ModuleList(
            [TransformerDecoderLayer(d_model=config.latent_dim, nhead=config.n_head) for _ in range(config.depth)]
        )      

    def forward(self, z, style, mask=None):
        '''
        input:
        z: [bs,32,sample_num,3]

        return:
        features: [bs,32,latent_dim]
        '''
        z = (z + self.pos_embedding).contiguous()
        for block in self.blocks:
            z = block(z,style)
        # features = torch.stack([self.local_decoders[i].decode(z[:,i]) for i in range(32)], dim=1)
        return z


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()

    # test encoder
    encoder = Encoder(40, 128, 3, 64)
    encoder_out = encoder(x)
    print('Encoder out shape:', encoder_out.shape)
