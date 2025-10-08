# ====== eFold Ablation 2 (Option-A, sequence-only trunk) =====================
# Intent: build the pair map z ONLY from the sequence stream (SequenceToPair).
# We DO NOT construct any pairwise modules that won't be used. This keeps
# DDP happy (no unused params) and preserves the (s, z) interface for the head.

import os, sys, typing as T
from contextlib import ExitStack
from collections import defaultdict

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange

from ..core.batch import Batch
from ..core.model import Model

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, ".."))

# -----------------------------------------------------------------------------#
#                               Top-level model                                #
# -----------------------------------------------------------------------------#

class eFold(Model):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        c_z: int,
        d_cnn: int,
        num_blocks: int,
        no_recycles: int,
        dropout: float = 0.0,
        lr: float = 1e-3,
        gamma: float = 0.995,
        loss_fn=nn.MSELoss(),
        optimizer_fn=torch.optim.Adam,
        **kwargs,
    ):
        self.save_hyperparameters(ignore=["loss_fn"])
        super().__init__(lr=lr, loss_fn=loss_fn, optimizer_fn=optimizer_fn, **kwargs)

        self.model_type = "eFold"
        self.data_type_output = ["structure"]
        self.lr = lr
        self.gamma = gamma
        self.loss = nn.MSELoss()

        # Sequence embedding (USED)
        self.encoder = nn.Embedding(ntoken, d_model)

        # IMPORTANT: remove unused pair adapter to avoid DDP unused params
        self.encoder_adapter = None  # not constructed in this ablation

        # Trunk: sequence-only; z is produced at the end from s
        self.eFold = EvoFoldAbl2(
            c_s=d_model,
            c_z=c_z,
            no_heads_s=8,
            num_blocks=num_blocks,
            dropout=dropout,
            no_recycles=no_recycles,
        )

        # Structure head (USED)
        self.structure_adapter = nn.Linear(c_z, d_cnn)
        self.output_structure = nn.Sequential(
            ResLayer(dim_in=d_cnn, dim_out=d_cnn // 2, n_blocks=4, kernel_size=3, dropout=dropout),
            ResLayer(dim_in=d_cnn // 2, dim_out=1, n_blocks=4, kernel_size=3, dropout=dropout),
        )

    def forward(self, batch: Batch) -> Tensor:
        src = batch.get("sequence")        # (B, L)
        s = self.encoder(src)              # (B, L, d_model)

        # Ablated trunk: returns (s, z) with z derived from final s
        s, z = self.eFold(s)               # z: (B, L, L, c_z)

        structure = self.structure_adapter(z)                       # (B, L, L, d_cnn)
        structure = self.output_structure(structure.permute(0,3,1,2)).squeeze(1)  # (B, L, L)
        return {"structure": (structure + structure.permute(0, 2, 1)) / 2}

    # Not used in this ablation; kept for API stability if other code imports it.
    def seq2map(self, seq_int):
        def int2seq(seq): return "".join(["XACGU"[d] for d in seq])

        def creatmat(data, device=None):
            with torch.no_grad():
                data = int2seq(data)
                paired = defaultdict(float, {"AU":2., "UA":2., "GC":3., "CG":3., "UG":0.8, "GU":0.8})
                mat = torch.tensor([[paired[x+y] for y in data] for x in data]).to(device)
                n = len(data)
                i, j = torch.meshgrid(torch.arange(n).to(device), torch.arange(n).to(device), indexing="ij")
                t = torch.arange(30).to(device)
                m1 = torch.where((i[:,:,None]-t>=0)&(j[:,:,None]+t<n),
                                 mat[torch.clamp(i[:,:,None]-t,0,n-1), torch.clamp(j[:,:,None]+t,0,n-1)], 0)
                m1 *= torch.exp(-0.5*t*t)
                m1_0pad = torch.nn.functional.pad(m1, (0,1))
                first0 = torch.argmax((m1_0pad==0).to(torch.int), dim=2)
                m1[t[None,None,:]>first0[:,:,None]] = 0
                m1 = m1.sum(dim=2)

                t = torch.arange(1,30).to(device)
                m2 = torch.where((i[:,:,None]+t<n)&(j[:,:,None]-t>=0),
                                 mat[torch.clamp(i[:,:,None]+t,0,n-1), torch.clamp(j[:,:,None]-t,0,n-1)], 0)
                m2 *= torch.exp(-0.5*t*t)
                m2_0pad = torch.nn.functional.pad(m2, (0,1))
                first0 = torch.argmax((m2_0pad==0).to(torch.int), dim=2)
                m2[torch.arange(29).to(device)[None,None,:]>first0[:,:,None]] = 0
                m2 = m2.sum(dim=2)
                m2[m1==0] = 0
                return (m1+m2).to(self.device)

        full_map = []
        one_hot_embed = torch.zeros((5, 4), device=self.device); one_hot_embed[1:] = torch.eye(4)
        for seq in seq_int:
            seq_hot = one_hot_embed[seq].type(torch.long)
            pair_map = torch.kron(seq_hot, seq_hot).reshape(len(seq), len(seq), 16)
            energy_map = creatmat(seq)
            full_map.append(torch.cat((pair_map, energy_map.unsqueeze(-1)), dim=-1))
        return torch.stack(full_map).permute(0,3,1,2).contiguous()


# -----------------------------------------------------------------------------#
#                         Ablation-2 sequence-only trunk                        #
# -----------------------------------------------------------------------------#

class EvoFoldAbl2(nn.Module):
    """
    Sequence-only trunk:
      * update s through stacked EvoBlockAbl2
      * build z ONCE from the final s via SequenceToPair
      * no pairwise positional embedding / convs / MLPs / bias
    """
    def __init__(self, c_s: int, c_z: int, no_heads_s: int = 8,
                 num_blocks: int = 4, dropout: float = 0.0, no_recycles: int = 0):
        super().__init__()
        self.itters = no_recycles + 1
        self.blocks = nn.ModuleList([
            EvoBlockAbl2(c_s=c_s, no_heads_s=no_heads_s, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.s_norm = nn.LayerNorm(c_s)
        self.z_from_seq = SequenceToPair(c_s, c_z // 2, c_z)

    def forward(self, seq_feats: Tensor) -> T.Tuple[Tensor, Tensor]:
        s = seq_feats
        for it in range(self.itters):
            with ExitStack() if it == self.itters - 1 else torch.no_grad():
                s = self.s_norm(s)
                for block in self.blocks:
                    s = block(s)
        z = self.z_from_seq(s)  # (B, L, L, c_z)
        return s, z


class EvoBlockAbl2(nn.Module):
    """
    One sequence-only block:
      LN -> RelPos MHA -> Drop/Add -> FF -> Add -> Conv -> Add -> LN -> FF -> Add -> LN -> ResidueMLP
    """
    def __init__(self, c_s: int, no_heads_s: int, dropout: float = 0.0):
        super().__init__()
        assert c_s % no_heads_s == 0
        self.c_s = c_s

        self.layernorm = nn.LayerNorm(c_s)
        self.seq_attention = RelPositionMultiHeadAttention(
            num_heads=no_heads_s, head_size=int(c_s/no_heads_s), output_size=c_s
        )
        self.pos = PositionalEncoding(c_s, dropout)
        self.ln = nn.LayerNorm(c_s, eps=1e-12, elementwise_affine=True)

        self.drop = nn.Dropout(dropout)
        self.FF1 = FFMod(c_s, dropout=dropout)
        self.FF2 = FFMod(c_s, dropout=dropout)
        self.convMod = ConvModule(input_dim=c_s, dropout=dropout)
        self.ln_3 = nn.LayerNorm(c_s)
        self.ln_4 = nn.LayerNorm(c_s)
        self.mlp_seq = ResidueMLP(c_s, 2 * c_s, dropout=dropout)

    def forward(self, sequence_state: Tensor) -> Tensor:
        assert sequence_state.dim() == 3 and sequence_state.size(-1) == self.c_s

        y = self.layernorm(sequence_state)
        pe = self.pos(y)
        y = self.ln(y)
        y, _ = self.seq_attention([y, y, y, pe], bias=None)
        sequence_state = sequence_state + self.drop(y)

        sequence_state = sequence_state + self.FF1(sequence_state)
        sequence_state = sequence_state + self.convMod(sequence_state)
        sequence_state = self.ln_3(sequence_state)
        sequence_state = sequence_state + self.FF2(sequence_state)
        sequence_state = self.ln_4(sequence_state)
        sequence_state = self.mlp_seq(sequence_state)
        return sequence_state


# -----------------------------------------------------------------------------#
#                          Shared building blocks                              #
# -----------------------------------------------------------------------------#

class SequenceToPair(nn.Module):
    def __init__(self, sequence_state_dim, inner_dim, pairwise_state_dim):
        super().__init__()
        self.layernorm = nn.LayerNorm(sequence_state_dim)
        self.proj = nn.Linear(sequence_state_dim, inner_dim * 2, bias=True)
        self.o_proj = nn.Linear(2 * inner_dim, pairwise_state_dim, bias=True)
        torch.nn.init.zeros_(self.proj.bias)
        torch.nn.init.zeros_(self.o_proj.bias)

    def forward(self, sequence_state):
        assert sequence_state.dim() == 3
        s = self.layernorm(sequence_state)
        s = self.proj(s)
        q, k = s.chunk(2, dim=-1)
        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]
        x = torch.cat([prod, diff], dim=-1)
        x = self.o_proj(x)  # (B, L, L, c_z)
        return x


class ResidueMLP(nn.Module):
    def __init__(self, embed_dim, inner_dim, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, embed_dim),
            nn.Dropout(dropout),
        )
    def forward(self, x): return x + self.mlp(x)


class FFMod(nn.Module):
    def __init__(self, emb_dim, dropout=0.0, expand=2):
        super().__init__()
        self.lin = nn.Linear(emb_dim, emb_dim * expand)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.lin_2 = nn.Linear(emb_dim * expand, emb_dim)
    def forward(self, x):
        x = self.lin(x); x = self.act(x); x = self.drop(x); x = self.lin_2(x)
        return self.drop(x)


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
        **kwargs,
    ):
        super().__init__()
        self.adaptive_scale = adaptive_scale
        if not adaptive_scale:
            self.ln = nn.LayerNorm(input_dim, elementwise_affine=True)
        else:
            self.scale = nn.Parameter(torch.ones(input_dim))
            self.bias = nn.Parameter(torch.zeros(input_dim))

        self.pw_conv_1 = nn.Conv1d(input_dim, conv_expansion_rate * input_dim, kernel_size=1, bias=True)
        self.act1 = GLU() if conv_use_glu else nn.SiLU()
        # Keep your original override to GLU:
        self.act1 = GLU()

        self.dw_conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=conv_expansion_rate * input_dim,
            kernel_size=5, padding=5 // 2, groups=depth_multiplier, bias=True
        )
        self.bn = nn.BatchNorm1d(conv_expansion_rate * input_dim, momentum=0.985)
        self.act2 = nn.SiLU()
        self.pw_conv_2 = nn.Conv1d(conv_expansion_rate * input_dim, input_dim, kernel_size=1, bias=True)
        self.do = nn.Dropout(dropout)

    def forward(self, inputs, training=False, pad_mask=None, **kwargs):
        if not self.adaptive_scale:
            outputs = self.ln(inputs)
        else:
            outputs = inputs
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


class GLU(nn.Module):
    def forward(self, x): return x[:, : x.size(1) // 2] * torch.sigmoid(x[:, x.size(1) // 2 :])


class ResLayer(nn.Module):
    def __init__(self, n_blocks, dim_in, dim_out, kernel_size, dropout=0.0):
        super().__init__()
        self.res_blocks = nn.Sequential(
            *[
                ResBlock(
                    inplanes=dim_in,
                    planes=dim_in,
                    kernel_size=kernel_size,
                    dilation1=12 * (4 - i % 4),
                    dilation2=pow(2, (i % 4)),
                    dropout=dropout,
                )
                for i in range(n_blocks)
            ]
        )
        self.conv_output = nn.Conv2d(dim_in, dim_out, kernel_size=7, padding=3, bias=True)
    def forward(self, x: Tensor) -> Tensor:
        x = self.res_blocks(x)
        x = self.conv_output(x)
        return x


class ResBlock(nn.Module):
    expansion: int = 1
    def __init__(self, inplanes: int, planes: int, kernel_size=3,
                 dilation1: int = 1, dilation2: int = 1, dropout=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, dilation=dilation1, kernel_size=kernel_size)
        self.dropout = nn.Dropout(p=dropout)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation2, kernel_size=kernel_size)
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.bn1(x); out = self.relu1(out); out = self.conv1(out)
        out = self.dropout(out); out = self.relu2(out); out = self.conv2(out)
        out += identity
        return out


def conv3x3(in_planes: int, out_planes: int, dilation: int = 1, kernel_size=3) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=dilation, bias=False, dilation=dilation)


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
        return self.pe[: x.shape[1]]


# ---- attention (sequence side) ----

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, output_size=None,
                 dropout=0.0, use_projection_bias=True, return_attn_coef=True, **kwargs):
        super().__init__()
        if output_size is not None and output_size < 1:
            raise ValueError("output_size must be a positive number")
        self.head_size = head_size
        self.num_heads = num_heads
        self.output_size = output_size
        self.return_attn_coef = return_attn_coef
        self.dropout = nn.Dropout(dropout)

        self.query = nn.Linear(num_heads * head_size, num_heads * head_size, bias=False)
        self.key   = nn.Linear(num_heads * head_size, num_heads * head_size, bias=False)
        self.value = nn.Linear(num_heads * head_size, num_heads * head_size, bias=False)

        self.projection_kernel = nn.Parameter(torch.rand(num_heads, head_size, output_size) * 2 - 1)
        self.projection_bias = nn.Parameter(torch.rand(output_size) * 2 - 1) if use_projection_bias else None

    def call_qkv(self, query, key, value, training=False):
        query = self.query(query); B, T, _ = query.size(); query = query.view(B, T, self.num_heads, self.head_size)
        key   = self.key(key);     B, T, _ = key.size();   key   = key.view(B, T, self.num_heads, self.head_size)
        value = self.value(value); B, T, _ = value.size(); value = value.view(B, T, self.num_heads, self.head_size)
        return query, key, value

    def call_attention(self, query, key, value, logits, bias=None, training=False, mask=None):
        if mask is not None:
            if len(mask.size()) != len(logits.size()):
                mask = mask.unsqueeze(-3)
            logits += -1e9 * (1.0 - mask.float())
        if bias is not None:
            logits = logits + rearrange(bias, "... lq lk h -> ... h lq lk")
        attn_coef = F.softmax(logits, dim=-1)
        attn_coef_dropout = self.dropout(attn_coef)
        multihead_output = torch.einsum("...HNM,...MHI->...NHI", attn_coef_dropout, value)
        output = torch.einsum("...NHI,HIO->...NO", multihead_output, self.projection_kernel)
        if self.projection_bias is not None:
            output += self.projection_bias
        return output, attn_coef

    def forward(self, inputs, training=False, mask=None, **kwargs):
        query, key, value = inputs
        query, key, value = self.call_qkv(query, key, value, training=training)
        depth = torch.tensor(self.head_size, dtype=torch.float32)
        query /= torch.sqrt(depth)
        logits = torch.einsum("...NHO,...MHO->...HNM", query, key)
        output, attn_coef = self.call_attention(query, key, value, logits, training=training, mask=mask)
        return (output, attn_coef) if self.return_attn_coef else output


class RelPositionMultiHeadAttention(MultiHeadAttention):
    def __init__(self, kernel_sizes=None, strides=None, **kwargs):
        super().__init__(**kwargs)
        num_pos_features = self.num_heads * self.head_size
        self.pos_kernel = nn.Parameter(torch.rand(self.num_heads, num_pos_features, self.head_size) * 2 - 1)
        self.pos_bias_u = nn.Parameter(torch.zeros(self.num_heads, self.head_size))
        self.pos_bias_v = nn.Parameter(torch.zeros(self.num_heads, self.head_size))

    @staticmethod
    def relative_shift(x):
        x_shape = x.size()
        x = torch.cat([x, x.new_zeros(x_shape[0], x_shape[1], x_shape[2], 1)], dim=-1)
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
        output, attn_coef = self.call_attention(query, key, value, logits, training=training, mask=mask, bias=bias)
        return (output, attn_coef) if self.return_attn_coef else output
# ============================================================================#