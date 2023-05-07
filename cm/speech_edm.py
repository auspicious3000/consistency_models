import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model

from .denoise_wavenet import DenoiserConfig, DiffNet
from .fastspeech_encoder_nn import Wav2Vec2Config, TransformerEncoder

from typing import Optional

import logging
import time
import os
logger = logging.getLogger(__name__)

from fairseq.pdb import set_trace


@dataclass
class SpeechDiffConfig(FairseqDataclass):

    encoder_args: FairseqDataclass = Wav2Vec2Config()
    
    denoiser_args: FairseqDataclass = DenoiserConfig()
    
    num_classes: int = field(
        default=500
    )
    dim_spk_embed: int = field(
        default=256
    )
    use_fp16: bool = field(
        default=False
    )


class SpeechEDM(BaseFairseqModel):
    def __init__(self, cfg: SpeechDiffConfig):
        super().__init__()
        
        self.encoder = TransformerEncoder(cfg.encoder_args)
        self.denoiser = DiffNet(cfg.denoiser_args)
        
        self.num_classes = cfg.num_classes
        self.encoder_embed_dim = cfg.encoder_args.encoder_embed_dim
        self.dim_spk_embed = cfg.dim_spk_embed
        
        self.embed_token = nn.Embedding(self.num_classes + 2, self.encoder_embed_dim, 0)
        self.embed_spk = nn.Linear(self.dim_spk_embed, self.encoder_embed_dim)
        
        self.dtype = torch.float32 if cfg.use_fp16 else torch.float32
        
    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        #self.encoder.half()
        #self.denoiser.half()
        #self.embed_token.half()
        #self.embed_spk.half()
        pass
        
    def forward(self, input_, timesteps, labels, label_mask, spk_embs_src):
        
        spk_embs = self.embed_spk(spk_embs_src.to(self.dtype))
        
        label_embs = self.embed_token(labels)
        
        label_encode, _ = self.encoder(label_embs, label_mask)
        
        cond = label_encode + spk_embs.unsqueeze(1)
        cond = cond * (~label_mask).unsqueeze(-1)
        
        feats_ = self.denoiser(input_.to(self.dtype), 
                               timesteps, cond)
        
        return feats_