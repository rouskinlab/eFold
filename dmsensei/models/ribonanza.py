from torch import nn, tensor
import torch
from ..config import device, seq2int, START_TOKEN, END_TOKEN, PADDING_TOKEN
from ..core.model import Model
from torch.nn import init

global_gain = 0.1


class Convolutional(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.conv = nn.Conv2d(params["num_heads"], params["num_heads"], 3, padding=1)
        self.batch2d = nn.BatchNorm2d(params["num_heads"])
        self.gelu = nn.GELU()
        if params["use_se"]:
            self.se = SqueezeAndExcitation(params)
        else:
            self.gammas = nn.Parameter(torch.ones(1, params["num_heads"], 1, 1))

    def forward(self, structure):
        # all across the function: [batch_size, num_heads, seq_len, seq_len]
        x = self.conv(structure)
        x = self.batch2d(x)
        if self.params["use_se"]:
            x = self.se(x)
        x = self.gelu(x)
        x = x + structure
        if not self.params["use_se"]:
            x = x * self.gammas
        return x


class FeedForward(nn.Module):
    def __init__(
        self,
        params,
    ):
        super().__init__()
        self.params = params
        self.layer_norm = nn.LayerNorm(params["embed_dim"])
        self.linear1 = nn.Linear(params["embed_dim"], params["hidden_dim"])
        init.xavier_normal_(self.linear1.weight, gain=global_gain)
        self.gelu = torch.nn.GELU()
        self.linear2 = nn.Linear(params["hidden_dim"], params["embed_dim"])
        init.xavier_normal_(self.linear2.weight, gain=global_gain)

    def forward(self, sequence):
        self.layer_norm(sequence)
        sequence = self.linear1(sequence)
        sequence = self.gelu(sequence)
        sequence = self.linear2(sequence)
        return sequence


class SqueezeAndExcitation(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.adaptive_average_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(params["num_heads"], params["num_heads"])
        init.xavier_normal_(self.fc1.weight, gain=global_gain)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(params["num_heads"], params["num_heads"])
        init.xavier_normal_(self.fc2.weight, gain=global_gain)
        self.sigmoid = nn.Sigmoid()

    def forward(self, structure):
        # [batch_size, num_heads, seq_len, seq_len] -> [batch_size, num_heads, 1, 1]
        weights = self.adaptive_average_pooling(structure).squeeze(-1).squeeze(-1)
        weights = self.relu(self.fc1(weights))
        weights = self.sigmoid(self.fc2(weights))
        # multiply structure by weights
        # [batch_size, num_heads, seq_len, seq_len] x [batch_size, num_heads, 1, 1]
        return structure * weights.unsqueeze(-1).unsqueeze(-1)


from torch.nn.functional import multi_head_attention_forward


class SelfAttention(nn.Module):
    def __init__(
        self,
        params,
    ):
        super().__init__()
        self.conv = Convolutional(params)
        self.pos_encoding = DynamicPositionalEncoding(params)
        self.params = params

    def forward(self, sequence, structure):
        # define Q, K, V
        # [batch_size, seq_len, embed_dim] -> [batch_size, num_heads, seq_len, dim_per_head]
        Q = V = K = sequence.reshape(
            sequence.shape[0],
            sequence.shape[1],
            self.params["num_heads"],
            self.params["dim_per_head"],
        ).permute(0, 2, 1, 3)

        # blocks of attention map
        bpp_features = self.conv(structure)
        relative_positional_bias = self.pos_encoding(sequence)

        # attention map: [batch_size, num_heads, seq_len, seq_len]
        attention_map = (
            K @ Q.transpose(-1, -2) / (self.params["dim_per_head"] ** 0.5)
            + relative_positional_bias
            + bpp_features
        )

        # output: [batch_size, seq_len, embed_dim]
        output = (attention_map @ V).reshape(*sequence.shape)
        return output, bpp_features


class DynamicPositionalEncoding(nn.Module):
    def __init__(
        self,
        params,
    ):
        super().__init__()
        self.params = params
        self.positional_encoding = nn.Parameter(
            self.create_matrix(params["max_len"]).float(), requires_grad=False
        )
        self.lin1 = nn.Linear(1, 48)
        init.xavier_normal_(self.lin1.weight, gain=global_gain)
        self.silu = nn.SiLU()
        self.lin2 = nn.Linear(48, 48)
        init.xavier_normal_(self.lin2.weight, gain=global_gain)
        self.lin3 = nn.Linear(48, params["num_heads"])
        init.xavier_normal_(self.lin3.weight, gain=global_gain)

    def create_matrix(self, size):
        # Initialize an empty matrix filled with zeros
        matrix = torch.zeros((size, size), dtype=torch.int64)
        # Fill the upper diagonals with 1, 2, 3
        for i in range(size - 1):
            matrix[i, i + 1 :] = torch.arange(1, size - i)
        # Fill the lower diagonals with -1, -2, -3
        for i in range(1, size):
            matrix[i, :i] = torch.arange(-1, -i - 1, -1)
        return matrix

    def forward(self, sequence):
        seq_len = sequence.shape[1]
        # [seq_len, seq_len, 1]
        x = self.positional_encoding[:seq_len, :seq_len].unsqueeze(-1)
        # [seq_len, seq_len, 1] -> [seq_len, seq_len, 48]
        x = self.lin1(x)
        x = self.silu(x)
        x = self.lin2(x)
        x = self.silu(x)
        # [seq_len, seq_len, 48] -> [seq_len, seq_len, num_heads]
        x = self.lin3(x)
        # [seq_len, seq_len, num_heads] -> [1, num_heads, seq_len, seq_len]
        x = x.permute(2, 0, 1).unsqueeze(0)
        return x


class Preprocessing:
    def sequence_batch(batch):
        out = []
        L = max(batch.get("length"))
        for sequence, length in zip(batch.get("sequence"), batch.get("length")):
            out.append(
                torch.concat(
                    [
                        tensor([START_TOKEN], dtype=torch.long).to(device),
                        sequence[:length],
                        tensor([END_TOKEN], dtype=torch.long).to(device),
                        tensor([PADDING_TOKEN] * (L - length), dtype=torch.long).to(
                            device
                        ),
                    ],
                )
            )
        return torch.stack(out)

    def structure_batch(batch):
        structure = batch.get("structure")
        batch_size, L, _ = structure.shape
        embedded_matrix = torch.zeros(
            (batch_size, L + 2, L + 2), dtype=torch.float32
        ).to(device)
        embedded_matrix[:, 1:-1, 1:-1] = structure
        return embedded_matrix


class Encoder(nn.Module):
    def __init__(
        self,
        params,
    ):
        super().__init__()
        self.self_attention = SelfAttention(params)
        self.feed_forward = FeedForward(params)

    def forward(self, sequence, structure):
        encoded_sequence, structure = self.self_attention(sequence, structure)
        sequence = sequence + encoded_sequence
        sequence = self.feed_forward(sequence) + sequence
        return sequence, structure


class Ribonanza(Model):
    ntokens = 7
    data_type = ["dms", "shape"]

    def __init__(
        self,
        params,
        lr=1e-3,
        optimizer_fn=torch.optim.Adam,
    ):
        super().__init__(**params, lr=lr, optimizer_fn=optimizer_fn)

        # Layers
        self.table_embedding = nn.Embedding(self.ntokens, params["embed_dim"])
        init.xavier_uniform_(self.table_embedding.weight, gain=global_gain)
        self.encoders_stack = nn.Sequential(
            *[Encoder(params) for _ in range(params["num_encoders"])]
        )
        self.output_net = nn.Linear(params["embed_dim"], 2)
        init.xavier_normal_(self.output_net.weight, gain=global_gain)

        # params
        self.params = params
        params["table_embedding"] = self.table_embedding

    def forward(self, batch):
        # Preprocessing
        sequence = Preprocessing.sequence_batch(batch)
        structure = Preprocessing.structure_batch(batch)
        # Embedding
        sequence = self.table_embedding(sequence)  # .reshape(sequence.shape[0], -1)
        structure = torch.unsqueeze(structure, 1).expand(
            -1, self.params["num_heads"], -1, -1
        )  # repeat for heads
        # Encoder
        for encoder in self.encoders_stack:
            sequence, structure = encoder(sequence, structure)
        # Output
        x = self.output_net(sequence)
        return {"dms": x[:, 1:-1, 0], "shape": x[:, 1:-1, 1]}
