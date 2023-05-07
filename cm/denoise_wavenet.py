import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.dataclass import FairseqDataclass
from dataclasses import dataclass, field

from math import sqrt

from fairseq.pdb import set_trace


@dataclass
class DenoiserConfig(FairseqDataclass):
    
    input_dim: int = field(
        default=80,
        metadata={"help": "spectrogram dim"},
    )
    residual_channels: int = field(
        default=512,
    )
    encoder_hidden: int = field(
        default=512,
    )
    residual_layers: int = field(
        default=40,
    )
    dilation_cycle_length: int = field(
        default=2,
    )
    dropout: float = field(
        default=0.1,
    )

    
def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')    
    

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x
    
    
class PositionalEmbedding_R(nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=True):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.sin(), x.cos()], dim=1)
        return x


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class ResidualBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation, dropout):
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = Linear(residual_channels, residual_channels)
        self.conditioner_projection = Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)
        self.dropout = dropout

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1).to(x.dtype)
        conditioner = self.conditioner_projection(conditioner)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        y = x + diffusion_step
        
        y = self.dilated_conv(y) + conditioner

        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) * sqrt(0.5), skip


class DiffNet(nn.Module):
    def __init__(self, cfg: DenoiserConfig):
        super().__init__()
        
        self.in_dims = cfg.input_dim
        self.residual_channels = cfg.residual_channels
        self.encoder_hidden = cfg.encoder_hidden
        self.dilation_cycle_length = cfg.dilation_cycle_length
        self.residual_layers = cfg.residual_layers
        self.dropout = cfg.dropout
        
        self.input_projection = Conv1d(self.in_dims, self.residual_channels, 1)
        self.diffusion_embedding = PositionalEmbedding_R(self.residual_channels)
        
        self.mlp = nn.Sequential(
            Linear(self.residual_channels, self.residual_channels * 4),
            nn.Mish(),
            Linear(self.residual_channels * 4, self.residual_channels),
            nn.Mish()
        )
        
        self.residual_layers = nn.ModuleList([
            ResidualBlock(self.encoder_hidden, 
                          self.residual_channels, 
                          2 ** (i % self.dilation_cycle_length),
                          self.dropout)
            for i in range(self.residual_layers)
        ])
        
        self.skip_projection = Conv1d(self.residual_channels, self.residual_channels, 1)
        self.output_projection = Conv1d(self.residual_channels, self.in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        
        cond = cond.transpose(1, 2)
        x = spec.transpose(1, 2)
        x = self.input_projection(x)  # x [B, residual_channel, T]
        x = F.mish(x)
        
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        
        skip = []
        for layer_id, layer in enumerate(self.residual_layers):
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) * sqrt(1.0 / len(self.residual_layers))
        x = F.mish(x)
        x = self.skip_projection(x)
        x = F.mish(x)
        x = self.output_projection(x)  # [B, 80, T]
        
        return x.transpose(1, 2)
