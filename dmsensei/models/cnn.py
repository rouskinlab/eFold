import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoderLayer
import numpy as np
import os
import sys
from ..core.model import Model
from ..core.batch import Batch
from einops import rearrange
import torch.nn.functional as F

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, ".."))


class CNN(Model):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        d_cnn: int,
        n_heads: int,
        dropout: float = 0.1,
        lr: float = 1e-5,
        optimizer_fn=torch.optim.Adam,
        **kwargs,
    ):
        super().__init__(lr=lr, optimizer_fn=optimizer_fn, **kwargs)

        self.model_type = "CNN"
        self.data_type_output = ["structure"]
        self.d_model = d_model
        self.d_cnn = d_cnn

        self.encoder = nn.Embedding(ntoken, d_model)
        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder_adapter = nn.Linear(d_model, d_cnn // 2)
        # self.structure_adapter = nn.Linear(d_cnn // 8, n_heads)
        self.activ = nn.ReLU()

        self.res_layers = nn.Sequential(
            ResLayer(
                dim_in=2 * d_cnn // 2,
                dim_out=d_cnn // 2,
                n_blocks=3,
                kernel_size=3,
                dropout=dropout,
            ),
            ResLayer(
                dim_in=d_cnn // 2,
                dim_out=d_cnn // 4,
                n_blocks=6,
                kernel_size=3,
                dropout=dropout,
            ),
            ResLayer(
                dim_in=d_cnn // 4,
                dim_out=d_cnn // 8,
                n_blocks=4,
                kernel_size=3,
                dropout=dropout,
            ),
        )

        self.output_structure = ResLayer(
            dim_in=d_cnn // 8, dim_out=1, n_blocks=3, kernel_size=3, dropout=dropout
        )

        # self.seq_attention1 = Attention(d_model, n_heads, d_model // n_heads)
        # self.seq_attention2 = Attention(d_model, n_heads, d_model // n_heads)

        # self.output_net_DMS = nn.Sequential(
        #     nn.Linear(d_model, d_model * 2),
        #     nn.LayerNorm(d_model * 2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(d_model * 2, d_model),
        #     nn.LayerNorm(d_model),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(d_model, 1),
        # )

        # self.output_net_SHAPE = nn.Sequential(
        #     nn.Linear(d_model, d_model * 2),
        #     nn.LayerNorm(d_model * 2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(d_model * 2, d_model),
        #     nn.LayerNorm(d_model),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(d_model, 1),
        # )

    def forward(self, batch: Batch) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = batch.get("sequence")
        src = self.encoder(src)
        # src = self.pos_encoder(src)

        x = self.activ(self.encoder_adapter(src))  # (N, L, d_cnn/2)

        # Outer concatenation
        matrix = x.unsqueeze(1).repeat(1, x.shape[1], 1, 1)  # (N, L, L, d_cnn/2)
        matrix = torch.cat(
            (matrix, matrix.permute(0, 2, 1, 3)), dim=-1
        )  # (N, L, L, d_cnn)

        # Resnet layers
        matrix = self.res_layers(matrix.permute(0, 3, 1, 2))  # (N, d_cnn//8, L, L)

        # Output structure
        structure = self.output_structure(matrix).squeeze(1)  # (N, L, L)

        # matrix = self.activ(
        #     self.structure_adapter(matrix.permute(0, 2, 3, 1))
        # )  # (N, L, L, d_model)

        # # Output DMS/SHAPE sequences
        # seq = self.seq_attention1(src, matrix)  # (N, L, d_model)
        # seq = self.seq_attention2(seq, matrix)  # (N, L, d_model)

        # dms = self.output_net_DMS(seq)
        # shape = self.output_net_SHAPE(seq)

        return {
            # "dms": dms.squeeze(dim=-1),
            # "shape": shape.squeeze(dim=-1),
            "structure": (structure + structure.permute(0, 2, 1)) / 2,
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


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, head_width):
        super(Attention, self).__init__()
        assert embed_dim == num_heads * head_width

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_width = head_width

        self.proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.rescale_factor = self.head_width**-0.5

        torch.nn.init.zeros_(self.o_proj.bias)

    def forward(self, x, bias=None):
        """
        Basic self attention with optional mask and external pairwise bias.
        To handle sequences of different lengths, use mask.

        Inputs:
          x: batch of input sequneces (.. x L x C)
          bias: batch of scalar pairwise attention biases (.. x Lq x Lk x num_heads). optional.

        Outputs:
          sequence projection (B x L x embed_dim), attention maps (B x L x L x num_heads)
        """

        t = rearrange(self.proj(x), "... l (h c) -> ... h l c", h=self.num_heads)
        q, k, v = t.chunk(3, dim=-1)

        q = self.rescale_factor * q
        a = torch.einsum("...qc,...kc->...qk", q, k)

        # Add external attention bias.
        if bias is not None:
            a = a + rearrange(bias, "... lq lk h -> ... h lq lk")

        a = F.softmax(a, dim=-1)

        y = torch.einsum("...hqk,...hkc->...qhc", a, v)
        y = rearrange(y, "... h c -> ... (h c)", h=self.num_heads)

        y = self.o_proj(y)

        return y  # , rearrange(a, "... lq lk h -> ... h lq lk")


class ResLayer(nn.Module):
    def __init__(self, n_blocks, dim_in, dim_out, kernel_size, dropout=0.0):
        super(ResLayer, self).__init__()

        # Basic Residula block
        self.res_layers = []
        for i in range(n_blocks):
            # dilation = pow(2, (i % 3))
            self.res_layers.append(
                ResBlock(
                    inplanes=dim_in,
                    planes=dim_in,
                    kernel_size=kernel_size,
                    dilation1=12 * (4 - i % 4),
                    dilation2=pow(2, (i % 4)),
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
        self.conv2 = conv3x3(
            planes, planes, dilation=dilation2, kernel_size=kernel_size
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
