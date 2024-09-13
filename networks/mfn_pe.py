import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('../')
from util import sigma2alpha

sys.path.append('../')

class MFNBase(nn.Module):
    """
    Multiplicative filter network base class.

    Expects the child class to define the 'filters' attribute, which should be
    a nn.ModuleList of n_layers+1 filters with output equal to hidden_size.
    """

    def __init__(
            self, hidden_size, out_size, n_layers, weight_scale, bias=True, output_act=False
    ):
        super().__init__()

        self.linear = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias) for _ in range(n_layers)]
        )
        self.output_linear = nn.Linear(hidden_size, out_size)
        self.output_act = output_act

        for lin in self.linear:
            lin.weight.data.uniform_(
                -np.sqrt(weight_scale / hidden_size),
                np.sqrt(weight_scale / hidden_size),
            )
        
        #self.initialize_weights()
        
        return
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.filters[0](x)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.linear[i - 1](out)
        out = self.output_linear(out)

        if self.output_act:
            out = torch.sin(out)
        


        return out

# Embedder class definition
class Embedder:

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        #self.embed_fns_array = embed_fns
        self.embed_fns_array = [torch.vmap(fn) for fn in embed_fns]
        self.out_dim = out_dim

    def embed(self, inputs):

      result_tensor = torch.cat([fn(inputs) for fn in self.embed_fns_array], dim=-1)
      return result_tensor

def get_embedder(multires=10, i=0):    # 10 is default value for multires

    if i == -1:
        return lambda x: x, 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj):
      return eo.embed(x)
    return embed, embedder_obj.out_dim


# GaborNet class definition
class GaborLayerPE(nn.Module):
    """
    Gabor-like filter as used in GaborNet.
    """

    def __init__(self, in_features, out_features, weight_scale, alpha=1.0, beta=1.0):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.mu = nn.Parameter(2 * torch.rand(out_features, in_features) - 1)
        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,))
        )

        self.linear.weight.data *= weight_scale * torch.sqrt(self.gamma[:, None])
        self.linear.bias.data.uniform_(-np.pi, np.pi)

        return

    def forward(self, x):
        D = (
                (x ** 2).sum(-1)[..., None]
                + (self.mu ** 2).sum(-1)[None, :]
                - 2 * x @ self.mu.T
        )
        return torch.sin(self.linear(x)) * torch.exp(-0.5 * D * self.gamma[None, :])


class GaborNetPE(MFNBase):
    def __init__(
            self,
            in_size,            # 3D coordinates
            hidden_size,        # size of hidden layer
            out_size,
            n_layers=3,         # number of layers
            input_scale=256.0,
            weight_scale=1.0,
            alpha=6.0,
            beta=1.0,
            bias=True,
            output_act=False,
            embed_kwargs=None  # Embedder kwargs
    ):
        super().__init__(
            hidden_size, out_size, n_layers, weight_scale, bias, output_act
        )
        self.embed_obj, self.input_ch = get_embedder()

        self.in_size = self.input_ch

        self.filters = nn.ModuleList(
            [
                GaborLayerPE(
                    self.in_size,
                    #n_layers,
                    hidden_size,
                    input_scale / np.sqrt(n_layers + 1),
                    alpha / (n_layers + 1),
                    beta,
                )
                for _ in range(n_layers + 1)
            ]
        )

    def forward(self, x):
        # Embedding
        x_embed = self.embed_obj(x)

        out = self.filters[0](x_embed)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x_embed) * self.linear[i - 1](out)

        out = self.output_linear(out)

        if self.output_act:
            out = torch.sin(out)

        return out

    def gradient(self, x):
        # Gradient computation
        x.requires_grad_(True)
        y = self.forward(x)[..., -1:]
        y = F.softplus(y - 1.)
        y = sigma2alpha(y)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)
