import os
import sys

sys.path.append(os.path.abspath("."))
import wandb
import pandas as pd
from dmsensei.config import device
from dmsensei import DataModule, create_model
import sys
import os

import tqdm
import torch 
import time


dm = DataModule(
    name=["yack_train"], # finetune: "utr", "pri_miRNA", "archiveII"
    strategy='random', #random, sorted or ddp
    num_workers=10,
    shuffle_train=False,
    data_type=["dms", "shape", "structure"],  #
    force_download=False,
    batch_size=1,
    max_len=1024,
    structure_padding_value=0,
    train_split=None,
    external_valid=["yack_valid", "utr", "pri_miRNA", "human_mRNA"], # finetune: "yack_valid", "human_mRNA"
)
dm.setup('fit')

model = create_model(
    model="cnn",
    ntoken=5,
    d_model=64,
    d_cnn=128,
    n_heads=16,
    dropout=0,
    lr=1e-4,
    weight_decay=0,
    gamma=0.995,
).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([300])).to(device)
wandb.init(project="debugging", name="test dm")
timings = []
for epoch in range(100):    
    tic = time.time()
    for idx, batch in enumerate(dm.train_dataloader()):#, desc="Epoch {}".format(epoch), total=len(dm.train_dataloader())):
        batch.to(device)
        pred = model.forward(batch)
        batch.integrate_prediction(pred)
        pred, true = batch.get_pairs("structure")
        loss = loss_fn(pred, true)
        optimizer.step()
        optimizer.zero_grad()
        del batch
        timings.append(time.time() - tic)
        tic = time.time()
        if idx % 100 == 0:
            torch.cuda.empty_cache()
            print("Mean time per batch: {}".format(sum(timings) / len(timings)))
            wandb.log({"mean_time_per_batch_ms": 1000 * sum(timings) / len(timings)})
            timings = []