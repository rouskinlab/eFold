import sys
sys.path.append('../../eFold')
from efold import inference
import pandas as pd
import numpy as np
from rouskinhf import get_dataset
import torch

import time
from tqdm import tqdm

from rnastructure_wrapper import RNAstructure

Fold = RNAstructure(path='/root/RNAstructure/exe/')

rnaStructure_dTs = []
efold_GPU_dTs = []
efold_CPU_dTs = []

lengths = np.linspace(10, 1000, 100).astype(int)
# lengths = [2000]
for length in tqdm(lengths):
    
    sequence_random = ''.join(np.random.choice(['A', 'C', 'G', 'U'], length))

    # eFold GPU
    t0 = time.time()
    inference(sequence_random, fmt='bp', user_device='cuda')
    dT = time.time()-t0
    efold_GPU_dTs.append(dT)

    # RNAfold
    t0 = time.time()
    inference(sequence_random, fmt='bp', user_device='cpu')
    dT = time.time()-t0
    efold_CPU_dTs.append(dT)

    # RNAstructure no MFE
    t0 = time.time()
    Fold.fold(sequence_random, mfe_only=False)
    dT = time.time()-t0
    rnaStructure_dTs.append(dT)

import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=lengths, y=rnaStructure_dTs, mode='lines', name='RNAstructure'))
fig.add_trace(go.Scatter(x=lengths, y=efold_CPU_dTs, mode='lines', name='eFold (CPU)'))
fig.add_trace(go.Scatter(x=lengths, y=efold_GPU_dTs, mode='lines', name='eFold (GPU)'))
fig.update_layout(title='Inference time of eFold vs RNAstructure', 
                  xaxis_title='Sequence length', yaxis_title='Time [s]', 
                  template='plotly_white', font_size=22, margin=dict(l=100, r=20, t=100, b=100),
                  width=2000, height=1200)
# fig.show()

fig.write_image('speed_comparison.jpg')

