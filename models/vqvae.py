
import torch
import torch.nn as nn
import numpy as np
from models.encoder_decoder import Encoder, Decoder
from models.quantizer import VectorQuantizer
from models.pointnet_plus_encoder import PointNetEncoder
from utils.builder import MODELS 
from einops import rearrange
from loguru import logger
from utils.pointcloud import write_pointcloud
from models.PCN import PCN

@MODELS.register_module()
class VQVAE(nn.Module):
    def __init__(self, config):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.local_ae = nn.ModuleList([PointNetEncoder(config.encoder) for _ in range(32)])
        self.encoder = Encoder(config.encoder)
        self.style_encoders = nn.ModuleList([PCN(config.style_encoder) for _ in range(32)])
        self.vector_quantization = VectorQuantizer(
            config.quantizer)
        # decode the discrete latent representation
        self.decoder = Decoder(config.decoder)
        if config.args.ae_checkpoint != '':
            checkpoint = torch.load(config.args.ae_checkpoint)
            logger.info('----loading ae checkpoints----')
            for i in range(32):
                self.local_ae[i].load_state_dict(checkpoint['base_model'])
                # self.local_ae[i].requires_grad_(False)

    def forward(self, x, masks=None):
        centered_points = x
        features = torch.stack([self.local_ae[i].encode(centered_points[:,i]) for i in range(32)],dim=1)
        # features.masked_fill_((masks<0.5).unsqueeze(2).repeat(1,1,features.shape[2]),0)
        z_e = self.encoder(features,masks<0.5)
        style_encoding = torch.stack([self.style_encoders[i].encode(x[:,i]) for i in range(32)],dim=1)
        # embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(
        #     z_e)
        embedding_loss = torch.tensor([0.]).cuda()
        z_q = z_e
        z_q = rearrange(z_q,'b c w h -> b (w h) c')
        x_hat = self.decoder(z_q, style_encoding)
        x_hat = torch.stack([self.local_ae[i].decode(x_hat[:,i]) for i in range(32)], dim=1)
        return embedding_loss, x_hat
