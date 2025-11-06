import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Dict
from collections import defaultdict

from ..core.batch import Batch
from ..core.model import Model


class TRM(Model):
    def __init__(
        self,
        ntoken: int,
        c_z: int = 32,
        k_steps: int = 4,
        z_cycles: int = 3,
        dropout: float = 0.0,
        lr: float = 3e-4,
        gamma: float = 0.995,
        loss_fn=nn.BCEWithLogitsLoss,
        optimizer_fn=torch.optim.Adam,
        **kwargs,
    ):
        self.save_hyperparameters(ignore=["loss_fn"])
        super().__init__(lr=lr, optimizer_fn=optimizer_fn, **kwargs)

        self.model_type = "TRM"
        self.data_type_output = ["structure"]
        self.lr = lr
        self.gamma = gamma

        # Pairwise input from sequence-only priors (17 channels from seq2map)
        in_pair_channels = 17
        self.pair_encoder = nn.Sequential(
            nn.Conv2d(in_pair_channels, c_z, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=max(1, c_z // 8), num_channels=c_z),
            nn.GELU(),
        )

        # Tiny recursive update of latent z given current answer y (as a channel)
        z_update_in = c_z + 1  # concatenate current y logits map
        self.z_update = nn.Sequential(
            nn.Conv2d(z_update_in, c_z, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=max(1, c_z // 8), num_channels=c_z),
            nn.GELU(),
            nn.Conv2d(c_z, c_z, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=max(1, c_z // 8), num_channels=c_z),
            nn.GELU(),
        )

        # Head to propose improvements to y from z
        self.y_head = nn.Sequential(
            nn.Conv2d(c_z, c_z, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(c_z, 1, kernel_size=1, bias=True),
        )

        self.c_z = c_z
        self.k_steps = k_steps
        self.z_cycles = z_cycles
        self.dropout = dropout

    def forward(self, batch: Batch) -> Dict[str, Tensor]:
        # Build initial pairwise representation z from sequence-only priors
        seq_int = batch.get("sequence")  # (N, L)
        pair_priors = self.seq2map(seq_int)  # (N, 17, L, L)
        z = self.pair_encoder(pair_priors)  # (N, c_z, L, L)

        # Initialize logits answer map y
        N, _, L, _ = z.shape
        y = torch.zeros((N, 1, L, L), device=z.device, dtype=z.dtype)

        # Recursive improvement steps
        for _ in range(self.k_steps):
            # Recursively refine latent z conditioned on current y
            for _ in range(self.z_cycles):
                z = z + self.z_update(torch.cat([z, y], dim=1))
            # Update answer logits from refined z (residual)
            y = y + self.y_head(z)

        # Symmetrize logits to respect i<->j pairing symmetry
        y_logits = y.squeeze(1)  # (N, L, L)
        y_logits = 0.5 * (y_logits + y_logits.transpose(1, 2))
        return {"structure": y_logits}

    def seq2map(self, seq_int: Tensor) -> Tensor:
        """
        Convert integer-encoded sequences to a pairwise feature map with:
        - 16 channels from Kronecker product of one-hot nucleotides
        - 1 channel of heuristic pairing energy prior
        Returns (N, 17, L, L)
        """

        def int2seq(seq):
            return "".join(["XACGU"[d] for d in seq])

        def creatmat(data, device=None):
            with torch.no_grad():
                data = int2seq(data)
                paired = defaultdict(float, {"AU": 2.0, "UA": 2.0, "GC": 3.0, "CG": 3.0, "UG": 0.8, "GU": 0.8})
                mat = torch.tensor([[paired[x + y] for y in data] for x in data]).to(device)
                n = len(data)
                i, j = torch.meshgrid(torch.arange(n).to(device), torch.arange(n).to(device), indexing="ij")
                t = torch.arange(30).to(device)
                m1 = torch.where(
                    (i[:, :, None] - t >= 0) & (j[:, :, None] + t < n),
                    mat[torch.clamp(i[:, :, None] - t, 0, n - 1), torch.clamp(j[:, :, None] + t, 0, n - 1)],
                    0,
                )
                m1 *= torch.exp(-0.5 * t * t)
                m1_0pad = F.pad(m1, (0, 1))
                first0 = torch.argmax((m1_0pad == 0).to(torch.int), dim=2)
                to0indices = t[None, None, :] > first0[:, :, None]
                m1[to0indices] = 0
                m1 = m1.sum(dim=2)

                t = torch.arange(1, 30).to(device)
                m2 = torch.where(
                    (i[:, :, None] + t < n) & (j[:, :, None] - t >= 0),
                    mat[torch.clamp(i[:, :, None] + t, 0, n - 1), torch.clamp(j[:, :, None] - t, 0, n - 1)],
                    0,
                )
                m2 *= torch.exp(-0.5 * t * t)
                m2_0pad = F.pad(m2, (0, 1))
                first0 = torch.argmax((m2_0pad == 0).to(torch.int), dim=2)
                to0indices = torch.arange(29).to(device)[None, None, :] > first0[:, :, None]
                m2[to0indices] = 0
                m2 = m2.sum(dim=2)
                m2[m1 == 0] = 0
                return (m1 + m2).to(self.device)

        # Assemble all channels: 16 Kronecker one-hot + 1 energy prior
        full_map = []
        one_hot_embed = torch.zeros((5, 4), device=self.device)
        one_hot_embed[1:] = torch.eye(4, device=self.device)
        for seq in seq_int:
            seq_hot = one_hot_embed[seq].type(torch.long)
            pair_map = torch.kron(seq_hot, seq_hot).reshape(len(seq), len(seq), 16)
            energy_map = creatmat(seq)
            full_map.append(torch.cat((pair_map, energy_map.unsqueeze(-1)), dim=-1))

        return torch.stack(full_map).permute(0, 3, 1, 2).contiguous()


