import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoderLayer
import numpy as np
import os, sys, json
from scipy.stats.stats import pearsonr

import lightning.pytorch as pl

from ..templates import DMSModel
import wandb

import sys, os

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, ".."))

# from util_torch import *


class Transformer(DMSModel):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.1,
        lr: float = 1e-2,
        loss_fn=nn.MSELoss(),
        optimizer_fn=torch.optim.Adam,
        **kwargs,
    ):
        super().__init__(lr=lr, loss_fn=loss_fn, optimizer_fn=optimizer_fn, **kwargs)

        self.d_model = d_model
        self.nhead = nhead
        self.nlayers = nlayers
        self.model_type = "Transformer"
        self.encoder = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        '''
        self.transformer_encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model,
                    nhead,
                    d_hid,
                    dropout,
                    batch_first=True,
                    norm_first=True,
                    activation="gelu",
                )
                for i in range(nlayers)
            ]
        )
        '''
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderLayer(d_model,nhead,d_hid,dropout,batch_first=True,norm_first=True,activation="gelu",)for i in range(nlayers)])
        # self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.decoder = nn.Linear(d_model, 1)
        # self.output_train_end = None
        initrange = 1
        self.encoder.weight.data.uniform_(0, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(0, initrange)

        self.sigmoid = nn.Sigmoid()
        # self.init_weights()
        self.output_net = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.LayerNorm(d_model*2),
            nn.ReLU(inplace=True),
            #nn.Flatten(start_dim=1, end_dim=-1),  
            nn.Linear(d_model*2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),  
            nn.Linear(d_model, 1)
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight*0.001)
                m.bias.data.fill_(0.01)

        self.output_net.apply(init_weights)
        # self.train_losses = []
        # self.val_losses = []
        # self.val_corr = []
        # self.epoch_count = 0
        self.sigmoid = nn.Sigmoid()

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """

        padding_mask = src == 0
        src = self.encoder(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        for i, l in enumerate(self.transformer_encoder):
            src = self.transformer_encoder[i](src, src_key_padding_mask=padding_mask)

        # output = self.output_net(src)
        # output = self.sigmoid(torch.flatten(output, start_dim=1))
        #print(src.flatten(start_dim=1, end_dim=-1).size())
        return self.output_net(src).flatten(start_dim=1, end_dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.shape[1]]
        # x = torch.concatenate((x, torch.tile(self.pe[:x.shape[1]], (x.shape[0], 1, 1))), 2)
        return self.dropout(x)
