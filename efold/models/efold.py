import numpy as np
import torch
from torch import nn, Tensor
import os
import sys
from contextlib import ExitStack

import typing as T
from einops import rearrange
import torch.nn.functional as F
import numpy as np
from ..core.batch import Batch
from ..core.model import Model

from collections import defaultdict    

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, ".."))


class eFold(Model):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        c_z: int,
        d_cnn: int,
        num_blocks: int,
        no_recycles: int,
        dropout: float = 0,
        lr: float = 1e-2,
        gamma: float = 1.0,
        loss_fn=nn.MSELoss(),
        optimizer_fn=torch.optim.Adam,
        **kwargs,
    ):
        self.save_hyperparameters(ignore=['loss_fn'])
        super().__init__(
            lr=lr, loss_fn=loss_fn, optimizer_fn=optimizer_fn, **kwargs
        )

        self.model_type = "eFold"
        self.data_type_output = ["structure"]
        self.lr = lr
        self.gamma = gamma
        self.train_losses = []
        self.loss = nn.MSELoss()

        # Encoder layers
        self.encoder = nn.Embedding(ntoken, d_model)
        # self.encoder_adapter = nn.Linear(d_model, int(c_z / 2))
        # self.activ = nn.ReLU()
        self.encoder_adapter = nn.Conv2d(17, c_z, kernel_size=15, padding=7, bias=True)
        self.eFold = EvoFold(
            c_s=d_model,
            c_z=c_z,
            # CHANGE
            no_heads_s=8,
            ######
            no_heads_z=4,
            num_blocks=num_blocks,
            dropout=dropout,
            no_recycles=no_recycles,
        )

        # self.output_net_DMS = nn.Sequential(
        #     nn.LayerNorm(d_model),
        #     nn.Linear(d_model, d_model),
        #     nn.ReLU(),
        #     nn.Linear(d_model, 1),
        # )

        # self.output_net_SHAPE = nn.Sequential(
        #     nn.LayerNorm(d_model),
        #     nn.Linear(d_model, d_model),
        #     nn.ReLU(),
        #     nn.Linear(d_model, 1),
        # )

        self.structure_adapter = nn.Linear(c_z, d_cnn)
        self.output_structure = nn.Sequential(
            ResLayer(
                dim_in=d_cnn,
                dim_out=d_cnn // 2,
                n_blocks=4,
                kernel_size=3,
                dropout=dropout,
            ),
            ResLayer(
                dim_in=d_cnn // 2, dim_out=1, n_blocks=4, kernel_size=3, dropout=dropout
            ),
        )

    def forward(self, batch: Batch) -> Tensor:
        # Encoding of RNA sequence
        src = batch.get("sequence")
        
        s = self.encoder(src)  # (N, L, d_model)
        z = self.encoder_adapter(self.seq2map(src)).permute(0, 2, 3, 1) # (N, L, L, d_model)

        # z = self.activ(self.encoder_adapter(s))  # (N, L, c_z / 2)
        # # Outer concatenation
        # z = z.unsqueeze(1).repeat(1, z.shape[1], 1, 1)  # (N, L, L, c_z / 2)
        # z = torch.cat((z, z.permute(0, 2, 1, 3)), dim=-1)  # (N, L, L, c_z)

        s, z = self.eFold(s, z)

        structure = self.structure_adapter(z)  # (N, L, L, d_cnn)
        structure = self.output_structure(structure.permute(0, 3, 1, 2)).squeeze(
            1
        )  # (N, L, L)

        return {
            # "dms": self.output_net_DMS(s).squeeze(axis=2),
            # "shape": self.output_net_SHAPE(s).squeeze(axis=2),
            "structure": (structure + structure.permute(0, 2, 1))
            / 2,
        }
        
    def seq2map(self, seq_int):

        def int2seq(seq):
            # return ''.join(['XAUCG'[d] for d in seq])
            return ''.join(['XACGU'[d] for d in seq])

        # take integer encoded sequence and return last channel of embedding (pairing energy)
        def creatmat(data, device=None):

            with torch.no_grad():
                data = int2seq(data)
                paired = defaultdict(float, {'AU':2., 'UA':2., 'GC':3., 'CG':3., 'UG':0.8, 'GU':0.8})

                mat = torch.tensor([[paired[x+y] for y in data] for x in data]).to(device)
                n = len(data)

                i, j = torch.meshgrid(torch.arange(n).to(device), torch.arange(n).to(device), indexing='ij')
                t = torch.arange(30).to(device)
                m1 = torch.where((i[:, :, None] - t >= 0) & (j[:, :, None] + t < n), mat[torch.clamp(i[:,:,None]-t, 0, n-1), torch.clamp(j[:,:,None]+t, 0, n-1)], 0)
                m1 *= torch.exp(-0.5*t*t)

                m1_0pad = torch.nn.functional.pad(m1, (0, 1))
                first0 = torch.argmax((m1_0pad==0).to(int), dim=2)
                to0indices = t[None,None,:]>first0[:,:,None]
                m1[to0indices] = 0
                m1 = m1.sum(dim=2)

                t = torch.arange(1, 30).to(device)
                m2 = torch.where((i[:, :, None] + t < n) & (j[:, :, None] - t >= 0), mat[torch.clamp(i[:,:,None]+t, 0, n-1), torch.clamp(j[:,:,None]-t, 0, n-1)], 0)
                m2 *= torch.exp(-0.5*t*t)

                m2_0pad = torch.nn.functional.pad(m2, (0, 1))
                first0 = torch.argmax((m2_0pad==0).to(int), dim=2)
                to0indices = torch.arange(29).to(device)[None,None,:]>first0[:,:,None]
                m2[to0indices] = 0
                m2 = m2.sum(dim=2)
                m2[m1==0] = 0

                return (m1+m2).to(self.device)

        # Assemble all data
        full_map = []
        one_hot_embed = torch.zeros((5, 4), device=self.device)
        one_hot_embed[1:] = torch.eye(4)
        for seq in seq_int:

            seq_hot = one_hot_embed[seq].type(torch.long)
            pair_map = torch.kron(seq_hot, seq_hot).reshape(len(seq), len(seq), 16)

            energy_map = creatmat(seq)

            full_map.append(torch.cat((pair_map, energy_map.unsqueeze(-1)), dim=-1))


        return torch.stack(full_map).permute(0, 3, 1, 2).contiguous()


class EvoBlock(nn.Module):
    def __init__(
        self,
        c_s,
        c_z,
        no_heads_s,
        no_heads_z,
        dropout=0,
    ):
        super(EvoBlock, self).__init__()
        assert c_s % no_heads_s == 0
        assert c_z % no_heads_z == 0
        assert c_z % 2 == 0
        assert no_heads_s % 2 == 0
        assert no_heads_z % 2 == 0

        self.c_s = c_s
        self.c_z = c_z

        self.layernorm = nn.LayerNorm(c_s)

        # Adapter to add sequence rep to pair rep
        self.sequence_to_pair = SequenceToPair(c_s, c_z // 2, c_z)

        # bias attention heads
        self.pair_to_sequence = PairToSequence(c_z, no_heads_s)

        # self.seq_attention = Attention(c_s, no_heads_s, c_s / no_heads_s, gated=True)

        # print('---------')
        # print(no_heads_s)
        # print(int(c_s/no_heads_s))
        # print('---------')
        self.seq_attention = RelPositionMultiHeadAttention(
            num_heads=no_heads_s, head_size=int(c_s / no_heads_s), output_size=c_s
        )
        self.pos = PositionalEncoding(self.c_s, dropout)
        self.ln = nn.LayerNorm(self.c_s, eps=1e-12, elementwise_affine=True)

        self.resNet = ResLayer(
            dim_in=c_z, dim_out=c_z, n_blocks=2, kernel_size=3, dropout=dropout
        )

        # self.tri_mul_out = TriangleMultiplicationOutgoing(
        #     c_z,
        #     c_z,
        # )

        # self.tri_mul_in = TriangleMultiplicationIncoming(
        #     c_z,
        #     c_z,
        # )

        # self.tri_att_start = TriangleAttentionStartingNode(
        #     c_z,
        #     int(c_z / no_heads_z),
        #     no_heads_z,
        # )

        # self.tri_att_end = TriangleAttentionEndingNode(
        #     c_z,
        #     int(c_z / no_heads_z),
        #     no_heads_z,
        # )

        # Transition
        self.mlp_seq = ResidueMLP(c_s, 2 * c_s, dropout=dropout)
        self.mlp_pair = ResidueMLP(c_z, 2 * c_z, dropout=dropout)

        assert dropout < 0.4
        self.drop = nn.Dropout(dropout)
        # self.row_drop = Dropout(dropout * 2, 2)
        # self.col_drop = Dropout(dropout * 2, 1)

        self.FF1 = FFMod(c_s, dropout=dropout)
        self.FF2 = FFMod(c_s, dropout=dropout)
        self.convMod = ConvModule(input_dim=c_s, dropout=dropout)

        # self.ln_1 = nn.LayerNorm(c_s)
        # self.ln_2 = nn.LayerNorm(c_s)
        self.ln_3 = nn.LayerNorm(c_s)
        self.ln_4 = nn.LayerNorm(c_s)

        self._initZeros()

    def _initZeros(self):
        # torch.nn.init.zeros_(self.tri_mul_in.linear_z.weight)
        # torch.nn.init.zeros_(self.tri_mul_in.linear_z.bias)
        # torch.nn.init.zeros_(self.tri_mul_out.linear_z.weight)
        # torch.nn.init.zeros_(self.tri_mul_out.linear_z.bias)
        # torch.nn.init.zeros_(self.tri_att_start.mha.linear_o.weight)
        # torch.nn.init.zeros_(self.tri_att_start.mha.linear_o.bias)
        # torch.nn.init.zeros_(self.tri_att_end.mha.linear_o.weight)
        # torch.nn.init.zeros_(self.tri_att_end.mha.linear_o.bias)

        torch.nn.init.zeros_(self.sequence_to_pair.o_proj.weight)
        torch.nn.init.zeros_(self.sequence_to_pair.o_proj.bias)
        torch.nn.init.zeros_(self.pair_to_sequence.linear.weight)
        # torch.nn.init.zeros_(self.seq_attention.o_proj.weight)
        # torch.nn.init.zeros_(self.seq_attention.o_proj.bias)
        torch.nn.init.zeros_(self.mlp_seq.mlp[-2].weight)
        torch.nn.init.zeros_(self.mlp_seq.mlp[-2].bias)
        torch.nn.init.zeros_(self.mlp_pair.mlp[-2].weight)
        torch.nn.init.zeros_(self.mlp_pair.mlp[-2].bias)

    def forward(self, sequence_state, pairwise_state):
        """
        Inputs:
          sequence_state: B x L x sequence_state_dim
          pairwise_state: B x L x L x pairwise_state_dim

        Output:
          sequence_state: B x L x sequence_state_dim
          pairwise_state: B x L x L x pairwise_state_dim
        """
        assert len(sequence_state.shape) == 3
        assert len(pairwise_state.shape) == 4

        batch_dim, seq_dim, sequence_state_dim = sequence_state.shape
        pairwise_state_dim = pairwise_state.shape[3]
        assert sequence_state_dim == self.c_s
        assert pairwise_state_dim == self.c_z
        assert batch_dim == pairwise_state.shape[0]
        assert seq_dim == pairwise_state.shape[1]
        assert seq_dim == pairwise_state.shape[2]

        # Update sequence state
        bias = self.pair_to_sequence(pairwise_state)

        # Self attention with bias + mlp.
        y = self.layernorm(sequence_state)
        pe = self.pos(y)
        y = self.ln(y)
        # y, _ = self.seq_attention(y,bias=bias)
        y, _ = self.seq_attention([y, y, y, pe], bias=bias)
        sequence_state = sequence_state + self.drop(y)
        # FF + Local conv + FF

        sequence_state_ff = self.FF1(sequence_state)
        sequence_state = sequence_state + sequence_state_ff

        # sequence_stae = self.ln_2(sequence_state)
        sequence_state_con = self.convMod(sequence_state)
        sequence_state = sequence_state + sequence_state_con

        sequence_state = self.ln_3(sequence_state)
        sequence_state_ff = self.FF2(sequence_state)
        sequence_state = sequence_state + sequence_state_ff

        sequence_state = self.ln_4(sequence_state)

        sequence_state = self.mlp_seq(sequence_state)

        # Update pairwise state
        pairwise_state = pairwise_state + self.sequence_to_pair(sequence_state)

        pairwise_state = self.resNet(pairwise_state.permute(0, 3, 1, 2)).permute(
            0, 2, 3, 1
        )

        # # Axial attention
        # pairwise_state = pairwise_state + self.row_drop(
        #     self.tri_mul_out(pairwise_state)
        # )
        # pairwise_state = pairwise_state + self.col_drop(
        #     self.tri_mul_in(pairwise_state)
        # )
        # pairwise_state = pairwise_state + self.row_drop(
        #     self.tri_att_start(pairwise_state)
        # )
        # pairwise_state = pairwise_state + self.col_drop(
        #     self.tri_att_end(pairwise_state)
        # )

        # MLP over pairs.
        pairwise_state = self.mlp_pair(pairwise_state)

        return sequence_state, pairwise_state


class EvoFold(nn.Module):
    def __init__(
        self,
        c_s=1024,
        c_z=128,
        no_heads_s=8,
        no_heads_z=4,
        num_blocks=10,
        position_bins=32,
        dropout=0.0,
        no_recycles=2,
    ):
        super(EvoFold, self).__init__()
        """
        Inputs:
            c_z:            Channels for pair rep
            c_s:            Channels for seq rep
            position_bins:  Position Encoder Bins
        """

        # First 'recycle' is just the standard forward pass through the model.
        self.itters = no_recycles + 1

        self.pairwise_positional_embedding = RelativePosition(position_bins, c_z)

        self.blocks = nn.ModuleList(
            [
                EvoBlock(
                    c_s=c_s,
                    c_z=c_z,
                    no_heads_s=no_heads_s,
                    no_heads_z=no_heads_z,
                    dropout=dropout,
                )
                for i in range(num_blocks)
            ]
        )

        self.s_norm = nn.LayerNorm(c_s)
        self.z_norm = nn.LayerNorm(c_z)

    def forward(self, seq_feats, pair_feats):
        """
        Inputs:
            seq_feats:     B x L x c_s          tensor of sequence features
            pair_feats:    B x L x L x c_z      tensor of pair features
            no_recycles:   scalar               number of passes through trunk

        Output:
            pair_feats:    B x L x L x c_z      tensor of pair features
        """

        B = seq_feats.shape[0]
        L = seq_feats.shape[1]

        s = seq_feats
        z = pair_feats

        res_index = torch.arange(L, device=z.device).expand((B, L))

        def applyTrunk(s, z, res_index):
            z = z + self.pairwise_positional_embedding(res_index)

            for block in self.blocks:
                s, z = block(s, z)
            return s, z

        for itter in range(self.itters):
            # Only compute gradients on last itter
            with ExitStack() if itter == self.itters - 1 else torch.no_grad():
                s = self.s_norm(s)
                z = self.z_norm(z)

                s, z = applyTrunk(s, z, res_index)

        return s, z


class SequenceToPair(nn.Module):
    def __init__(self, sequence_state_dim, inner_dim, pairwise_state_dim):
        super(SequenceToPair, self).__init__()

        self.layernorm = nn.LayerNorm(sequence_state_dim)
        self.proj = nn.Linear(sequence_state_dim, inner_dim * 2, bias=True)
        self.o_proj = nn.Linear(2 * inner_dim, pairwise_state_dim, bias=True)

        # set bias to zero
        torch.nn.init.zeros_(self.proj.bias)
        torch.nn.init.zeros_(self.o_proj.bias)

    def forward(self, sequence_state):
        """
        Inputs:
          sequence_state: B x L x c_s

        Output:
          pairwise_state: B x L x L x c_z

        Intermediate state:
          B x L x L x 2*inner_dim
        """

        assert len(sequence_state.shape) == 3

        s = self.layernorm(sequence_state)
        s = self.proj(s)
        q, k = s.chunk(2, dim=-1)

        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]

        x = torch.cat([prod, diff], dim=-1)
        x = self.o_proj(x)

        return x


class Dropout(nn.Module):
    """
    Implementation of dropout with the ability to share the dropout mask
    along a particular dimension.
    """

    def __init__(self, r: float, batch_dim: T.Union[int, T.List[int]]):
        super(Dropout, self).__init__()

        self.r = r
        if isinstance(batch_dim, int):
            batch_dim = [batch_dim]
        self.batch_dim = batch_dim
        self.dropout = nn.Dropout(self.r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = list(x.shape)
        if self.batch_dim is not None:
            for bd in self.batch_dim:
                shape[bd] = 1
        return x * self.dropout(x.new_ones(shape))


class PairToSequence(nn.Module):
    def __init__(self, pairwise_state_dim, num_heads):
        super(PairToSequence, self).__init__()

        self.layernorm = nn.LayerNorm(pairwise_state_dim)
        self.linear = nn.Linear(pairwise_state_dim, num_heads, bias=False)

    def forward(self, pairwise_state):
        """
        Inputs:
          pairwise_state: B x L x L x pairwise_state_dim

        Output:
          pairwise_bias: B x L x L x num_heads
        """
        assert len(pairwise_state.shape) == 4
        z = self.layernorm(pairwise_state)
        pairwise_bias = self.linear(z)
        return pairwise_bias


class ResidueMLP(nn.Module):
    def __init__(self, embed_dim, inner_dim, dropout=0):
        super(ResidueMLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.mlp(x)


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, head_width, gated=False):
        super(Attention, self).__init__()
        assert embed_dim == num_heads * head_width

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_width = head_width

        self.proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.gated = gated
        if gated:
            self.g_proj = nn.Linear(embed_dim, embed_dim)
            torch.nn.init.zeros_(self.g_proj.weight)
            torch.nn.init.ones_(self.g_proj.bias)

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

        if self.gated:
            y = self.g_proj(x).sigmoid() * y
        y = self.o_proj(y)

        return y, rearrange(a, "... lq lk h -> ... h lq lk")


class RelativePosition(nn.Module):
    def __init__(self, bins, c_z):
        super(RelativePosition, self).__init__()
        self.bins = bins

        # Note an additional offset is used so that the 0th position
        # is reserved for masked pairs.
        self.embedding = torch.nn.Embedding(2 * bins + 2, c_z)

    def forward(self, residue_index):
        """
        Input:
          residue_index: B x L tensor of indices (dytpe=torch.long)

        Output:
          pairwise_state: B x L x L x c_z tensor of embeddings
        """

        assert residue_index.dtype == torch.long

        diff = residue_index[:, None, :] - residue_index[:, :, None]
        diff = diff.clamp(-self.bins, self.bins)
        diff = diff + self.bins + 1  # Add 1 to adjust for padding index.
        output = self.embedding(diff)
        return output


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
        pe = self.pe[: x.shape[1]]
        # x = torch.concatenate((x, torch.tile(self.pe[:x.shape[1]], (x.shape[0], 1, 1))), 2)
        return pe


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_size,
        output_size=None,
        dropout=0.0,
        use_projection_bias=True,
        return_attn_coef=True,
        **kwargs,
    ):
        super(MultiHeadAttention, self).__init__(**kwargs)

        if output_size is not None and output_size < 1:
            raise ValueError("output_size must be a positive number")

        self.head_size = head_size
        self.num_heads = num_heads
        self.output_size = output_size
        self.use_projection_bias = use_projection_bias
        self.return_attn_coef = return_attn_coef

        self.dropout = nn.Dropout(dropout)
        self._dropout_rate = dropout

        input_max = (self.num_heads * self.head_size) ** -0.5
        ### DOUBLE CHECK THIS CODE:
        self.query = nn.Linear(num_heads * head_size, num_heads * head_size, bias=False)
        self.key = nn.Linear(num_heads * head_size, num_heads * head_size, bias=False)
        self.value = nn.Linear(num_heads * head_size, num_heads * head_size, bias=False)
        ###########################

        self.projection_kernel = nn.Parameter(
            torch.rand(num_heads, head_size, output_size) * 2 - 1
        )

        if use_projection_bias:
            self.projection_bias = nn.Parameter(torch.rand(output_size) * 2 - 1)
        else:
            self.projection_bias = None

    def call_qkv(self, query, key, value, training=False):
        # Verify shapes
        if key.size(-2) != value.size(-2):
            raise ValueError(
                "the number of elements in 'key' must be equal to "
                "the same as the number of elements in 'value'"
            )

        # Linear transformations
        query = self.query(query)
        B, T, E = query.size()
        query = query.view(B, T, self.num_heads, self.head_size)

        key = self.key(key)
        B, T, E = key.size()
        key = key.view(B, T, self.num_heads, self.head_size)

        value = self.value(value)
        B, T, E = value.size()
        value = value.view(B, T, self.num_heads, self.head_size)

        return query, key, value

    def call_attention(
        self, query, key, value, logits, bias=None, training=False, mask=None
    ):
        # Mask = attention mask with shape [B, Tquery, Tkey] with 1 for positions we want to attend, 0 for masked
        if mask is not None:
            if len(mask.size()) < 2:
                raise ValueError("'mask' must have at least 2 dimensions")
            if query.size(-3) != mask.size(-2):
                raise ValueError(
                    "mask's second to last dimension must be equal to "
                    "the number of elements in 'query'"
                )
            if key.size(-3) != mask.size(-1):
                raise ValueError(
                    "mask's last dimension must be equal to the number of elements in 'key'"
                )

        # Apply mask
        if mask is not None:
            mask = mask.float()

            # Possibly expand on the head dimension so broadcasting works
            if len(mask.size()) != len(logits.size()):
                mask = mask.unsqueeze(-3)

            logits += -1e9 * (1.0 - mask)

        if bias is not None:
            logits = logits + rearrange(bias, "... lq lk h -> ... h lq lk")

        attn_coef = F.softmax(logits, dim=-1)

        # Attention dropout
        attn_coef_dropout = self.dropout(attn_coef)

        # Attention * value
        multihead_output = torch.einsum(
            "...HNM,...MHI->...NHI", attn_coef_dropout, value
        )

        # Run the outputs through another linear projection layer. Recombining heads
        # is automatically done.
        output = torch.einsum(
            "...NHI,HIO->...NO", multihead_output, self.projection_kernel
        )

        if self.projection_bias is not None:
            output += self.projection_bias

        return output, attn_coef

    def forward(self, inputs, training=False, mask=None, **kwargs):
        query, key, value = inputs

        query, key, value = self.call_qkv(query, key, value, training=training)

        # Scale dot-product, doing the division to either query or key
        # instead of their product saves some computation
        depth = torch.tensor(self.head_size, dtype=torch.float32)
        query /= torch.sqrt(depth)

        # Calculate dot product attention
        logits = torch.einsum("...NHO,...MHO->...HNM", query, key)

        output, attn_coef = self.call_attention(
            query, key, value, logits, training=training, mask=mask
        )

        if self.return_attn_coef:
            return output, attn_coef
        else:
            return output


class RelPositionMultiHeadAttention(MultiHeadAttention):
    def __init__(self, kernel_sizes=None, strides=None, **kwargs):
        super(RelPositionMultiHeadAttention, self).__init__(**kwargs)

        num_pos_features = self.num_heads * self.head_size
        input_max = (self.num_heads * self.head_size) ** -0.5
        self.pos_kernel = nn.Parameter(
            torch.rand(self.num_heads, num_pos_features, self.head_size) * 2 - 1
        )
        self.pos_bias_u = nn.Parameter(torch.zeros(self.num_heads, self.head_size))
        self.pos_bias_v = nn.Parameter(torch.zeros(self.num_heads, self.head_size))

    @staticmethod
    def relative_shift(x):
        x_shape = x.size()
        x = F.pad(x, (1, 0))

        x = x.view(x_shape[0], x_shape[1], x_shape[3] + 1, x_shape[2])
        x = x[:, :, 1:, :].view(x_shape)
        return x

    def forward(self, inputs, bias=None, training=False, mask=None, **kwargs):
        query, key, value, pos = inputs

        query, key, value = self.call_qkv(query, key, value, training=training)

        pos = torch.einsum("...MI,HIO->...MHO", pos, self.pos_kernel)

        query_with_u = query + self.pos_bias_u
        query_with_v = query + self.pos_bias_v

        logits_with_u = torch.einsum("...NHO,...MHO->...HNM", query_with_u, key)
        logits_with_v = torch.einsum("...NHO,...MHO->...HNM", query_with_v, pos)

        logits_with_v = self.relative_shift(logits_with_v)

        logits = logits_with_u + logits_with_v[:, :, :, : logits_with_u.size(3)]

        depth = torch.tensor(self.head_size, dtype=torch.float32)
        logits /= torch.sqrt(depth)

        output, attn_coef = self.call_attention(
            query, key, value, logits, training=training, mask=mask, bias=bias
        )

        if self.return_attn_coef:
            return output, attn_coef
        else:
            return output


class GLU(nn.Module):
    def __init__(self, name="glu"):
        super(GLU, self).__init__()

    def forward(self, x):
        return x[:, : x.size(1) // 2] * torch.sigmoid(x[:, x.size(1) // 2 :])


class ConvModule(nn.Module):
    def __init__(
        self,
        input_dim,
        kernel_size=31,
        dropout=0.0,
        depth_multiplier=1,
        conv_expansion_rate=2,
        conv_use_glu=False,
        adaptive_scale=False,
        name="conv_module",
        **kwargs,
    ):
        super(ConvModule, self).__init__()

        self.adaptive_scale = adaptive_scale
        if not adaptive_scale:
            self.ln = nn.LayerNorm(input_dim, elementwise_affine=True)
        else:
            self.scale = nn.Parameter(torch.ones(input_dim))
            self.bias = nn.Parameter(torch.zeros(input_dim))

        pw1_max = input_dim**-0.5
        dw_max = kernel_size**-0.5
        pw2_max = input_dim**-0.5

        self.pw_conv_1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=conv_expansion_rate * input_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        if conv_use_glu:
            self.act1 = GLU()
        else:
            self.act1 = nn.SiLU()
        self.act1 = GLU()

        self.dw_conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=conv_expansion_rate * input_dim,
            kernel_size=5,
            stride=1,
            padding=(5 // 2),
            groups=depth_multiplier,
            bias=True,
        )

        self.bn = nn.BatchNorm1d(conv_expansion_rate * input_dim, momentum=0.985)
        self.act2 = nn.SiLU()

        self.pw_conv_2 = nn.Conv1d(
            in_channels=conv_expansion_rate * input_dim,
            out_channels=input_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.do = nn.Dropout(dropout)
        self.res_add = nn.quantized.FloatFunctional()

    def forward(self, inputs, training=False, pad_mask=None, **kwargs):
        if not self.adaptive_scale:
            outputs = self.ln(inputs)
        # else:
        # scale = self.scale.view(1, 1, -1)
        # bias = self.bias.view(1, 1, -1)
        # outputs = inputs * scale + bias

        B, T, E = outputs.size()
        outputs = outputs.view(B, E, T)
        outputs = self.pw_conv_1(outputs)
        outputs = self.act1(outputs)
        outputs = self.dw_conv(outputs)
        outputs = self.bn(outputs)
        outputs = self.act2(outputs)
        outputs = self.pw_conv_2(outputs)
        outputs = outputs.view(B, T, E)
        outputs = self.do(outputs)

        return outputs


class FFMod(nn.Module):
    def __init__(self, emb_dim, dropout=0.0, expand=2):
        super(FFMod, self).__init__()
        self.lin = nn.Linear(emb_dim, emb_dim * expand)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.lin_2 = nn.Linear(emb_dim * expand, emb_dim)

    def forward(self, x):
        x = self.lin(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.lin_2(x)

        return self.drop(x)
