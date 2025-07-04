from typing import Dict, List, Optional, Tuple, Callable
import logging
import torch
import os
import argparse
import torch.nn as nn
import torch.nn.functional as F
from models.uvit import UViT
import math
import numpy as np
import yaml

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)
    return a

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule  
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999).to('cuda')

def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

class diffusion(nn.Module):
    def __init__(self,
        in_features,
        sampling_timesteps= 100,
        timesteps = 1000,
        eta = 1.0
        ):
        super(diffusion, self).__init__()
        self.in_features = in_features
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.betas = cosine_beta_schedule(self.timesteps).cuda()
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.model = UViT()
        self.eta = eta
        self.sqrt_alphas_cumprod =  torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1-self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        assert self.sampling_timesteps <= self.timesteps
    
    def set_device(self,device):
        self.device = device

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start).to(torch.float32).cuda()
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod.to(t.device), t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod.to(t.device), t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    
    @torch.no_grad()
    def ddim_sample(self, features,centroid=None):
        batch = features.shape[0]
        shape = (batch, 32, 9)
        total_timesteps, sampling_timesteps, eta = self.timesteps, self.sampling_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        skip = self.timesteps // self.sampling_timesteps
        seq = (np.linspace(
                   0, np.sqrt(self.timesteps * 0.8), self.sampling_timesteps
                   ) ** 2
                   )

        # x = torch.randn(shape)
        x = torch.randn(shape).cuda()
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(batch) * i).cuda()
            next_t = (torch.ones(batch) * j).cuda()
            at = compute_alpha(self.betas.to(t.device), t.long())
            at_next = compute_alpha(self.betas.to(t.device), next_t.long())
            xt = xs[-1].to(torch.float32)
            if centroid is None:
                x0_t = self.model(xt,t,features)
            else:
                x0_t = self.model(xt,t,features,centroid)
            et = (xt - x0_t * at.sqrt())/((1-at).sqrt())
            c1 = (
                self.eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next)

        return xs[-1]

    
    def forward(self, batched_inputs, features):
        if not self.training:
            results = self.ddim_sample(features)
            return results  
        else:
            bs = batched_inputs.shape[0]
            t = [torch.randint(0, 1000 , (1,)).long() for _ in range(bs)]
            t = torch.stack(t).squeeze(-1).cuda()
            batched_inputs = batched_inputs     # [bs,32,6]
            noise = torch.randn_like(batched_inputs).cuda()
            x_t = self.q_sample(batched_inputs,t,noise)
            x_t = x_t.to(torch.float32)
            predicted= self.model(x_t,t,features)
            return predicted