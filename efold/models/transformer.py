import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoderLayer
import numpy as np
import os
import sys
from ..core.model import Model
from ..core.batch import Batch

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, ".."))


class Transformer(Model):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.1,
        lr: float = 1e-5,
        c_z=16,
        optimizer_fn=torch.optim.Adam,
        **kwargs,
    ):
        super().__init__(lr=lr, optimizer_fn=optimizer_fn, **kwargs)

        self.d_model = d_model
        self.nhead = nhead
        self.nlayers = nlayers
        self.model_type = "Transformer"

        self.encoder = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder_adapter = nn.Linear(d_model, int(c_z / 2))
        self.activ = nn.ReLU()

        self.transformer_encoder = nn.Sequential(
            *[
                TransformerEncoderLayer(
                    d_model,
                    nhead,
                    d_hid,
                    dropout,
                    batch_first=True,
                    norm_first=True,
                    activation="relu",
                )
                for i in range(nlayers)
            ]
        )

        self.resnet = nn.Sequential(
            ResLayer(n_blocks=4, dim_in=1, dim_out=8, kernel_size=3, dropout=dropout),
            # ResLayer(n_blocks=4, dim_in=4, dim_out=8, kernel_size=3, dropout=dropout),
            # ResLayer(n_blocks=4, dim_in=8, dim_out=4, kernel_size=3, dropout=dropout),
            ResLayer(n_blocks=4, dim_in=8, dim_out=1, kernel_size=3, dropout=dropout),
        )

        self.output_net_DMS = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
        )

        self.output_net_SHAPE = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
        )

        assert c_z % 4 == 0, "c_z must be divisible by 4"
        assert c_z >= 8, "c_z must be greater than 8"
        self.output_net_structure = nn.Sequential(
            ResLayer(
                n_blocks=4, dim_in=c_z, dim_out=c_z // 2, kernel_size=3, dropout=dropout
            ),
            ResLayer(
                n_blocks=4,
                dim_in=c_z // 2,
                dim_out=c_z // 4,
                kernel_size=3,
                dropout=dropout,
            ),
            ResLayer(
                n_blocks=4, dim_in=c_z // 4, dim_out=1, kernel_size=3, dropout=dropout
            ),
        )

    def forward(self, batch: Batch) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = batch.get("sequence")
        src = self.encoder(src)
        src = self.pos_encoder(src)

        for i, l in enumerate(self.transformer_encoder):
            src = self.transformer_encoder[i](src)

        src = self.resnet(src.unsqueeze(dim=1)).squeeze(dim=1)

        dms = self.output_net_DMS(src)
        shape = self.output_net_SHAPE(src)

        # Outer concatenation
        src = self.activ(self.encoder_adapter(src))
        matrix = src.unsqueeze(1).repeat(1, src.shape[1], 1, 1)  # (N, d_cnn/2, L, L)
        matrix = torch.cat(
            (matrix, matrix.permute(0, 2, 1, 3)), dim=-1
        )  # (N, d_cnn, L, L)

        # Resnet layers
        pair_prob = self.output_net_structure(matrix.permute(0, 3, 1, 2)).squeeze(
            1
        )  # (N, L, L)

        # Symmetrize
        structure = (pair_prob + pair_prob.permute(0, 2, 1)) / 2  # (N, L, L)

        return {
            "dms": dms.squeeze(dim=-1),
            "shape": shape.squeeze(dim=-1),
            "structure": structure,
        }


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


## Resnet module definitions ##
class ResLayer(nn.Module):
    def __init__(self, n_blocks, dim_in, dim_out, kernel_size, dropout=0.0):
        super(ResLayer, self).__init__()

        # Basic Residula block
        self.res_layers = []
        for i in range(n_blocks):
            dilation = pow(2, (i % 3))
            self.res_layers.append(
                ResBlock(
                    inplanes=dim_in,
                    planes=dim_in,
                    kernel_size=kernel_size,
                    dilation1=8 * (4 - i % 4),
                    dilation2=pow(2, (i % 3)),
                    dropout=dropout,
                )
            )
        self.res_blocks = nn.Sequential(*self.res_layers)

        # Adapter to change depth
        self.conv_output = nn.Conv2d(
            dim_in, dim_out, kernel_size=7, padding=3, bias=True
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.res_blocks(x)
        x = self.conv_output(x)

        return x


class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        kernel_size=3,
        dilation1: int = 1,
        dilation2: int = 1,
        dropout=0.0,
    ) -> None:
        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(
            inplanes, planes, dilation=dilation1, kernel_size=kernel_size
        )
        self.dropout = nn.Dropout(p=dropout)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=5,
            padding=2 * dilation2,
            dilation=dilation2,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out += identity

        return out


def conv3x3(
    in_planes: int, out_planes: int, dilation: int = 1, kernel_size=3
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        padding=dilation,
        bias=False,
        dilation=dilation,
    )
