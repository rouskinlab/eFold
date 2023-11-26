from .path import Path
from huggingface_hub import snapshot_download
from .env import Env
from os.path import exists
import os
from ..core.datapoint import Datapoint
import json
import pickle
from tqdm import tqdm as tqdm_fn
from ..config import device

def download_dataset(name:str):
    path = Path(name=name)
    snapshot_download(
            repo_id='rouskinlab/' + name,
            repo_type="dataset",
            local_dir=path.get_main_folder(),
            token=Env.get_hf_token(),
            allow_patterns=["data.json"],
        )

def get_dataset(name:str, force_download=False, tqdm=True):
    """This function returns a list of datapoints, similar to what's in the data.json file."""
    
    path = Path(name=name)
    
    if force_download:
        os.system(f'rm -rf {path.get_main_folder()}')
    
    if not exists(path.get_data_json()):
        print("{}: Downloading dataset from HuggingFace Hub...".format(name))
        download_dataset(name)
        print("{}: Download complete. File saved at {}".format(name, path.get_data_json()))
    
    if not exists(path.get_data_pickle()):
        print("{}: Converting data.json to data.pkl...".format(name))
        data = json.load(open(path.get_data_json()))    
        
        # Load data.json into a list of datapoints
        parser = lambda x: tqdm_fn(x, desc=name, colour='red', total=len(data)) if tqdm else lambda x: x
        lod = [Datapoint.from_data_json_line(line) for line in parser(data.items())]
        print("{}: Saving data.pkl...".format(name))
        pickle.dump(lod, open(path.get_data_pickle(), 'wb'))
        return lod
    
    print("{}: Loading data.pkl...".format(name))
    return pickle.load(open(path.get_data_pickle(), 'rb'))