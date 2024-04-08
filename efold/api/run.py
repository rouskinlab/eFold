import os
from typing import List, Union
from ..models import create_model
import torch
from os.path import join, dirname
from ..core import batch
from ..core.embeddings import sequence_to_int
from ..core.postprocess import postprocess_new_nc as postprocess
import numpy as np
from ..util.format_conversion import convert_bp_list_to_dotbracket
    
# Load best model
model = create_model(
    model="efold",
    ntoken=5,
    d_model=64,
    c_z=32,
    d_cnn=64,
    num_blocks=4,
    no_recycles=0,
    dropout=0,
    lr=3e-4,
    weight_decay=0,
    gamma=0.995,
)
model.load_state_dict(torch.load(join(dirname(dirname(__file__)), "resources/efold_weights.pt")), strict=False)


def _load_sequences_from_fasta(fasta:str):
    with open(fasta, "r") as f:
        lines = f.readlines()
    sequences = []
    for line in lines:
        if line.startswith(">"):
            sequences.append("")
        else:
            sequences[-1] += line.strip()
    return sequences

def _predict_structure(model, sequence:str):
    seq = sequence_to_int(sequence).unsqueeze(0) 
    b = batch.Batch(
        sequence=seq,
        reference=[""],
        length=[len(seq)],
        L = len(seq),
        use_error=False,
        batch_size=1,
        data_types=["sequence"],
        dt_count={"sequence": 1})

    # predict the structure
    pred = model(b)
    structure = postprocess(pred['structure'], 
                            model.seq2oneHot(b.get('sequence')),
                            0.01, 0.1, 100, 1.6, True, 1.5).detach().numpy().round().astype(int)[0]

    # turn into 1-indexed base pairs
    return [(b,c) for b, c in (np.stack(np.where(np.triu(structure) == 1)) + 1).T]

def run(arg:Union[str, List[str]]=None, fmt="dotbracket"):
    """Runs the Efold API on the provided sequence or fasta file.
    
    Args:
        arg (str): The sequence or the list of sequences to run Efold on, or the path to a fasta file containing the sequences.  
        
    Returns:
        dict: A dictionary containing the sequences as keys and the predicted secondary structures as values.
        
    Examples:
    >>> from efold.api.run import run
    >>> structure = run("GGGAAAUCC") # this is awful, we need to remove the prints
    No scaling, use preLN
    Replace GLU with swish for Conv
    No scaling, use preLN
    Replace GLU with swish for Conv
    No scaling, use preLN
    Replace GLU with swish for Conv
    No scaling, use preLN
    Replace GLU with swish for Conv
    >>> assert structure == {'GGGAAAUCC': [(1, 9), (2, 8)]}, "Test failed: {}".format(structure)
    
    """
    assert fmt in ["dotbracket", "basepair", 'bp'], "Invalid format. Must be either 'dotbracket' or 'basepair'"
    # Check if the input is valid
    if not arg:
        raise ValueError("Either sequence or fasta must be provided")
    if any([key in arg for key in [".", "/", "\\"]]):
        if not os.path.exists(arg):
            raise ValueError("File not found")
        sequences = _load_sequences_from_fasta(arg)
    elif type(arg) == str:
        sequences = [arg]
    elif hasattr(arg, "__iter__") and all([isinstance(s, str) for s in arg]):
        sequences = arg
    else:
        raise ValueError("Either sequence or fasta must be provided")

    structures = []
    for seq in sequences:  
        structure = _predict_structure(model, seq)
        if fmt == "dotbracket":
            db_structure = convert_bp_list_to_dotbracket(structure, len(seq))
            if db_structure != None:
                structure = db_structure
        structures.append(structure)

    return {seq: structure for seq, structure in zip(sequences, structures)}


