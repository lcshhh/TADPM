import torch.nn as nn
import functools
class AdaGN(nn.Module):
    '''
    adaptive group normalization
    '''
    def __init__(self, ndim, cfg, n_channel):
        """
        ndim: dim of the input features 
        n_channel: number of channels of the inputs 
        ndim_style: channel of the style features 
        """
        super().__init__()
        style_dim = cfg.latent_pts.style_dim 
        init_scale = cfg.latent_pts.ada_mlp_init_scale 
        self.ndim = ndim 
        self.n_channel = n_channel
        self.style_dim = style_dim
        self.out_dim = n_channel * 2
        self.norm = nn.GroupNorm(8, n_channel)
        in_channel = n_channel 
        self.emd = dense(style_dim, n_channel*2, init_scale=init_scale)
        self.emd.bias.data[:in_channel] = 1
        self.emd.bias.data[in_channel:] = 0

    def __repr__(self):
        return f"AdaGN(GN(8, {self.n_channel}), Linear({self.style_dim}, {self.out_dim}))" 
        
    def forward(self, image, style):
        # style: B,D 
        # image: B,D,N,1 
        CHECK2D(style)
        style = self.emd(style)
        if self.ndim == 3: #B,D,V,V,V
            CHECK5D(image)
            style = style.view(style.shape[0], -1, 1, 1, 1) # 5D 
        elif self.ndim == 2: # B,D,N,1 
            CHECK4D(image) 
            style = style.view(style.shape[0], -1, 1, 1) # 4D 
        elif self.ndim == 1: # B,D,N
            CHECK3D(image) 
            style = style.view(style.shape[0], -1, 1) # 4D 
        else:
            raise NotImplementedError

        factor, bias = style.chunk(2, 1)
        result = self.norm(image)
        result = result * factor + bias  
        return result 
    
class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels, dim=1, cfg={}):

        assert(len(cfg) > 0), cfg
        super().__init__()
        if dim==1:
            conv = nn.Conv1d
        else:
            conv = nn.Conv2d
        bn = functools.partial(AdaGN, dim, cfg) 
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [out_channels]
        layers = []
        for oc in out_channels:
            layers.append(conv(in_channels, oc, 1)) 
            layers.append(bn(oc))
            layers.append(Swish()) 
            in_channels = oc
        self.layers = nn.ModuleList(layers)

    def forward(self, *inputs): 
        if len(inputs) == 1 and len(inputs[0]) == 4:
            # try to fix thwn SharedMLP is the first layer 
            inputs = inputs[0] 
        if len(inputs) == 1: 
            raise NotImplementedError 
        elif len(inputs) == 4:
            assert(len(inputs) == 4), 'input, style'
            x, _, _, style = inputs 
            for l in self.layers:
                if isinstance(l, AdaGN): 
                    x = l(x, style)
                else:
                    x = l(x) 
            return (x, *inputs[1:])
        elif len(inputs) == 2:
            x, style = inputs 
            for l in self.layers:
                if isinstance(l, AdaGN): 
                    x = l(x, style)
                else:
                    x = l(x) 
            return x 
        else:
            raise NotImplementedError 