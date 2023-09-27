import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
import os, sys, json
from scipy.stats.stats import pearsonr

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pandas as pd
import wandb

import sys, os


device = "cpu"
print("setting de")
if torch.cuda.is_available():
    device = "cuda"
    print("Using CUDA")


def reverse_one_hot(sequence):
    seq = []
    for j in sequence:
        if sum(j) == 0:
            char = 5
        else:
            char = np.where(j == 1)[0][0]
        seq.append(char)
    return seq


class dms_seq_dataset(torch.utils.data.Dataset):
    def __init__(self):
        # Load dataset from library
        # data = np.load('dms_seq_initial_data/processed/DMSseq_fibroblast_vivo_DA50.train.npz')
        # self.sequences, self.structures = import_structure(size=size, reload=True, save=False)
        # struct = data['struct'][100000:]
        # struct_true = data['struct_true'][:50000,:]
        # seqeunces = data['seq'][:50000,:,:]

        # struct_true[struct_true < 0] = 0
        # print(struct_true.shape)
        # print(seqeunces.shape)
        # list_seq = []
        import json

        # Opening JSON file
        f = open("dms_signals.json")
        # returns JSON object as
        # a dictionary
        data = json.load(f)

        alist = []
        for i in data["sequence"].keys():
            arrayToAppend = np.array(data["sequence"][i])
            arrayToAppend[arrayToAppend < 255] = 0
            # print(arrayToAppend)
            alist.append(arrayToAppend)
        arr_seq = np.stack(alist)

        alist = []
        for i in data["DMS"].keys():
            arrayToAppend = np.array(data["DMS"][i])
            alist.append(arrayToAppend)
        arr_dms = np.stack(alist)

        # i = 0
        # for x in seqeunces:
        #   print('Update %d' % i, end='\r')
        #   i +=1
        #  j = reverse_one_hot(x)
        # list_seq.append(j)
        # seqs = np.array(list_seq)
        arr_seq = np.load("processed_sequences.npy")
        arr_dms = np.load("processed_dms.npy")
        # Convert to torch tensors
        self.sequences = torch.Tensor(arr_seq).type(torch.int)
        self.structures = torch.Tensor(arr_dms).type(torch.long)

        # # Flatten the dimensions
        # self.sequences = torch.flatten(self.sequences.permute(0, 2, 1), start_dim=1)

        # if size < 50e3:
        #    self.full_load = True
        #    self.sequences = self.sequences.to(device)
        #    self.structures = self.structures.to(device)

        # else:
        #   self.full_load = False
        self.full_load = False

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.full_load:
            return (self.sequences[idx], self.structures[idx])
        else:
            return (self.sequences[idx].to(device), self.structures[idx].to(device))


class TransformerModel(pl.LightningModule):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.1,
        lr: float = 1e-2,
    ):
        super().__init__()

        self.lr = lr
        self.d_model = d_model
        self.loss = nn.MSELoss()
        # self.loss = nn.PoissonNLLLoss()
        self.save_hyperparameters()

        self.model_type = "Transformer"
        self.encoder = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = TransformerEncoderLayer(
            d_model,
            nhead,
            d_hid,
            dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
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
        # self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.decoder = nn.Linear(d_model, 1)
        self.output_train_end = None
        initrange = 1
        self.encoder.weight.data.uniform_(0, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(0, initrange)

        self.sigmoid = nn.Sigmoid()
        # self.init_weights()
        self.output_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.output_net.apply(init_weights)
        self.train_losses = []
        self.val_losses = []
        self.val_corr = []
        self.epoch_count = 0
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

        output = self.output_net(src)
        return self.sigmoid(torch.flatten(output, start_dim=1))

    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        prediction = self.forward(inputs)

        self.output_train_end = targets
        # TODO : change to -1000 for mask
        #       try to do 0 1 masking
        #       keep full len of inputs and targets
        a_c = torch.where((inputs == 1) | (inputs == 2))
        torch_labels_a_c = targets[a_c]
        torch_pred_np_a_c = prediction[a_c]
        loss = self.loss(torch_pred_np_a_c, torch_labels_a_c)
        self.train_losses.append(loss.item())
        return loss

    def on_train_epoch_end(self):
        print("last outputs")
        print("train-end")
        print(self.train_losses[-1])
        self.train_losses = []

    def on_validation_epoch_end(self):
        self.log("valid/loss", np.mean(self.val_losses))
        print("val-losses")
        print(self.val_losses[-1])

        if self.epoch_count == 1 and not os.path.isfile(
            "a_c_val_corr_tracking_hq_a_c.csv"
        ):
            # need to change hardcoded length to the size of valid_size/ valid_batch_size
            df = pd.DataFrame(
                0, index=np.arange(len(range(20))), columns=[self.epoch_count]
            )
            self.val_corr += ["NA"] * (10 - len(self.val_corr))
            df[self.epoch_count] = self.val_corr
            df.to_csv("a_c_val_corr_tracking_hq_a_c.csv")
        elif self.epoch_count == 0:
            print(self.epoch_count)
        elif self.epoch_count == 1 and os.path.isfile("a_c_val_corr_tracking_a_c.csv"):
            df = pd.read_csv("a_c_val_corr_tracking_hq_a_c.csv", index_col=0)
            self.epoch_count += 20
            self.val_corr += ["NA"] * (10 - len(self.val_corr))
            df[self.epoch_count] = self.val_corr
            df.to_csv("a_c_val_corr_tracking_hq_a_c.csv")
        else:
            df = pd.read_csv("a_c_val_corr_tracking_hq_a_c.csv", index_col=0)
            self.val_corr += ["NA"] * (10 - len(self.val_corr))
            df[self.epoch_count] = self.val_corr
            df.to_csv("a_c_val_corr_tracking_hq_a_c.csv")
        self.val_losses = []
        self.val_corr = []
        self.epoch_count += 1

    def validation_step(self, batch, batch_idx):
        # TODO : change to -1000 for mask
        #       try to do 0 1 masking
        #       keep full len of inputs and targets
        feature, label = batch
        prediction = self.forward(feature)

        seq = feature.detach().cpu().numpy()
        a_c = np.where((seq == 1) | (seq == 2))

        labels = label.detach().cpu().numpy()
        pred_np = prediction.detach().cpu().numpy()

        labels_a_c = labels[a_c]
        pred_np_a_c = pred_np[a_c]
        a_c = torch.where((feature == 1) | (feature == 2))
        torch_labels_a_c = label[a_c]
        torch_pred_np_a_c = prediction[a_c]
        # Compute loss
        # loss = self.loss(prediction, label)
        loss = self.loss(torch_pred_np_a_c, torch_labels_a_c)
        self.val_losses.append(loss.item())
        self.val_corr.append(pearsonr(labels_a_c, pred_np_a_c)[0])
        return prediction, loss

    def test_step(self, batch, batch_idx):
        feature, label = batch

        prediction = self.forward(feature)
        loss = self.loss(prediction, label)

        seq = feature.detach().cpu().numpy()
        a_c = np.where((seq == 1) | (seq == 2))

        labels = label.detach().cpu().numpy()
        pred_np = prediction.detach().cpu().numpy()

        labels_a_c = labels[a_c]
        pred_np_a_c = pred_np[a_c]
        self.val_corr.append(pearsonr(labels_a_c, pred_np_a_c)[0])

        df = pd.DataFrame(
            0,
            index=np.arange(len(range(len(self.val_corr)))),
            columns=[self.epoch_count],
        )
        df["test_results"] = self.val_corr
        df.to_csv("a_c_val_corr_tracking_sars_test.csv")

        return prediction

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        return optimizer


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


'''
class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, nstruct: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, batch_first = True):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model, padding_idx=0)
        self.decoder = nn.Linear(d_model, 1)
        self.d_model = d_model

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]

        Returns:
            output Tensor of shape [batch_size, seq_len, ntoken]
        """
        padding_mask = src == 0
        src = self.encoder(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        #print(src.shape)
        output = self.transformer_encoder(src, src_key_padding_mask=padding_mask)
        #print(output.shape)
        output = self.decoder(output)
        return torch.flatten(output, start_dim=1)
'''
