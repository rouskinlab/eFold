import numpy as np
import torch
from torch import nn, Tensor
import lightning.pytorch as pl
import os, sys
from contextlib import ExitStack

import typing as T
from einops import rearrange
import torch.nn.functional as F
import numpy as np

from ..templates import DMSModel

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, ".."))

class Evoformer(DMSModel):

    def __init__(self,
                    ntoken: int,
                    d_model: int,
                    c_z: int,
                    num_blocks: int,
                    no_recycles: int,
                    dropout: float = 0,
                    lr: float = 1e-2,
                    gamma: float = 1.0,
                    loss_fn=nn.MSELoss(),
                    optimizer_fn=torch.optim.Adam,
                    **kwargs):
        self.save_hyperparameters()
        super(Evoformer, self).__init__(lr=lr, loss_fn=loss_fn, optimizer_fn=optimizer_fn, gamma=gamma, **kwargs)

        self.model_type = 'Evoformer'
        self.lr = lr
        self.gamma = gamma
        self.train_losses = []
        self.loss = nn.MSELoss()

        # Encoder layers
        self.encoder = nn.Embedding(ntoken, d_model)
        self.encoder_adapter = nn.Linear(d_model, int(c_z/2))
        self.activ = nn.ReLU()
        self.evoformer = EvoFold(
            c_s = d_model,
            c_z = c_z,
            no_heads_s = 32,
            no_heads_z = 4,
            num_blocks = num_blocks,
            dropout=dropout,
            no_recycles=no_recycles,
        )

        self.output_net_DMS = nn.Sequential(nn.LayerNorm(d_model),
                                                nn.Linear(d_model, d_model), 
                                                nn.ReLU(),
                                                nn.Linear(d_model, 1))

        self.output_net_SHAPE = nn.Sequential(nn.LayerNorm(d_model),
                                                nn.Linear(d_model, d_model), 
                                                nn.ReLU(),
                                                nn.Linear(d_model, 1))


    def forward(self, src: Tensor) -> Tensor:
        # Encoding of RNA sequence
        # (N, L, d_model)
        s = self.encoder(src)     

        # (N, L, c_z / 2)
        z = self.activ(self.encoder_adapter(s))   
        # Outer concatenation
        # (N,  c_z / 2, L, L)
        z = z.unsqueeze(1).repeat(1,z.shape[1],1,1)                
        # (N, c_z, L, L)
        z = torch.cat((z, z.permute(0,2,1,3)), dim=-1)

        s = self.evoformer(s, z)

        # z = self.output(z.permute(0,3,1,2)).squeeze(1)
        
        # Symmetrize
        # z = (z + z.permute(0,2,1)) / 2.0

        # output = self.output_adapter(s)
        # return output.squeeze(-1)
    
        DMS = self.output_net_DMS(s)
        SHAPE = self.output_net_SHAPE(s)
        return torch.concatenate((DMS, SHAPE), dim=-1)




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
        self.sequence_to_pair = SequenceToPair(
            c_s, c_z // 2, c_z
        )

        # bias attention heads
        self.pair_to_sequence = PairToSequence(c_z, no_heads_s)

        self.seq_attention = Attention(
            c_s, no_heads_s, c_s / no_heads_s, gated=True
        )

        self.resNet = ResLayer(dim_in=c_z, dim_out=c_z, n_blocks=2, kernel_size=3, dropout=dropout)

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
        self.mlp_seq = ResidueMLP(c_s, 4 * c_s, dropout=dropout)
        self.mlp_pair = ResidueMLP(c_z, 4 * c_z, dropout=dropout)

        assert dropout < 0.4
        self.drop = nn.Dropout(dropout)
        # self.row_drop = Dropout(dropout * 2, 2)
        # self.col_drop = Dropout(dropout * 2, 1)

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
        torch.nn.init.zeros_(self.seq_attention.o_proj.weight)
        torch.nn.init.zeros_(self.seq_attention.o_proj.bias)
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
        y, _ = self.seq_attention(y, bias=bias)
        sequence_state = sequence_state + self.drop(y)
        sequence_state = self.mlp_seq(sequence_state)

        # Update pairwise state
        pairwise_state = pairwise_state + self.sequence_to_pair(sequence_state)

        pairwise_state = self.resNet(pairwise_state.permute(0, 3, 1, 2)).permute(0,2,3,1)

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
    def __init__(self,
                  c_s = 1024,
                  c_z = 128,
                  no_heads_s = 32,
                  no_heads_z = 4,
                  num_blocks = 10, 
                  position_bins = 32,
                  dropout = 0.0,
                  no_recycles = 2):
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
                    c_s = c_s,
                    c_z = c_z,
                    no_heads_s = no_heads_s,
                    no_heads_z = no_heads_z,
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

        return s


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
        if type(batch_dim) == int:
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
            self.res_layers.append(ResBlock(inplanes=dim_in, planes=dim_in, kernel_size=kernel_size,
                                            dilation1=12*(4 - i%4), dilation2=pow(2, (i % 4)), dropout=dropout))
        self.res_blocks = nn.Sequential(*self.res_layers)

        # Adapter to change depth
        self.conv_output = nn.Conv2d(dim_in, dim_out, kernel_size=7, padding=3, bias=True)

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
        kernel_size = 3,
        dilation1: int = 1,
        dilation2: int = 1,
        dropout=0.0
        ) -> None:
        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, dilation=dilation1, kernel_size=kernel_size)
        self.dropout = nn.Dropout(p=dropout)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation2, kernel_size=kernel_size)

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

def conv3x3(in_planes: int, out_planes: int, dilation: int = 1, kernel_size=3) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                     padding=dilation, bias=False, dilation=dilation)
