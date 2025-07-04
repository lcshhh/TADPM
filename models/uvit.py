import torch
import torch.nn as nn
import math
import timm
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Mlp
import einops
import torch.utils.checkpoint

# the xformers lib allows less memory, faster training and inference
# try:
#     import xformers
#     import xformers.ops
#     XFORMERS_IS_AVAILBLE = True
#     print('xformers enabled')
# except:
XFORMERS_IS_AVAILBLE = False
print('xformers disabled')


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def patchify(imgs, patch_size):
    x = einops.rearrange(imgs, 'B C (h p1) (w p2) -> B (h w) (p1 p2 C)', p1=patch_size, p2=patch_size)
    return x


def unpatchify(x, channels=3):
    patch_size = 2
    h = 1
    w = x.shape[1] // h
    # w = x.shape[1] // h
    # h = w = int(x.shape[1] ** .5)
    # assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
    # x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h, p1=patch_size, p2=patch_size)
    x = einops.rearrange(x, 'B w (p C) -> B C (w p)', p=patch_size)
    return x

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=6, patch_size=2, in_chans=1, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        # self.img_size = img_size
        # self.patch_size = patch_size
        self.grid_size = int(img_size/patch_size)
        self.num_patches = self.grid_size
        self.flatten = flatten

        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # B, C, H, W = x.shape
        # assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        # x = x.unsqueeze(1)
        print(1,x.shape)
        x = self.proj(x)
        print(2,x.shape)
        if self.flatten:
            # x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if False: # XFORMERS_IS_AVAILBLE:  # the xformers lib allows less memory, faster training and inference
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        else:
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class UViT(nn.Module):
    def __init__(self, img_size=6, patch_size=2, in_chans=16, embed_dim=27, depth=12, num_heads=9, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, use_checkpoint=False,
                 num_clip_token=16, conv=True, skip=True, use_ae=False):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.in_chans = in_chans
        self.use_ae = use_ae
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        # self.context_embed = nn.Linear(clip_dim, 6)

        self.context_embed = nn.Sequential(
                nn.Linear(1827, 1024),
                nn.GELU(),
                nn.Linear(1024, 512),
                nn.GELU(),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, 9),
        )

        self.mlp = nn.Sequential(
                nn.Linear(64, 9),
                nn.GELU(),
                nn.Linear(9, 9),
        )

        self.extras = 1 + num_clip_token

        # self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 32, 27))

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)
        self.patch_dim = patch_size ** 2 * in_chans
        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias=True)
        self.final_layer = nn.LayerNorm(9)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x, timesteps, context):
        '''
        x:[bs,32,6]
        '''
        x = x.float()
        B, L, D = x.shape

        time_token = self.time_embed(timestep_embedding(timesteps, 9*32))
        time_token = time_token.unsqueeze(dim=1).view(B,32,9)
        context_token = self.context_embed(context)
        x = torch.cat((time_token, context_token, x), dim=2)
        x = x + self.pos_embed

        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        x = self.mid_block(x)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        x = self.norm(x)
        x = self.decoder_pred(x)
        x = self.mlp(x)
        x = self.final_layer(x)

        return x
