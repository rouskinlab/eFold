from .path import Path
from huggingface_hub import HfApi
import os
import datetime
import json
from .env import Env
from os.path import dirname
import numpy as np
from io import BytesIO

def name_from_path(datapath:str):
    if datapath.split('/')[-1] == 'data.json':
        return datapath.split('/')[-2]
    return datapath.split('/')[-1].split('.')[0]

def clean_data(datapath:str):   
    data = json.load(open(datapath, 'r'))
    for ref, values in data.items():
        copy = values.copy()
        for k, v in values.items():
            if v == None or (type(v) == float and np.isnan(v)):
                copy.pop(k) 
        data[ref] = copy
    return data

def upload_dataset(datapath:str, exist_ok= False, commit_message:str=None, add_card=True, **kwargs):
    api = HfApi()
    name = name_from_path(datapath)
    # data = clean_data(datapath)

    hf_token = Env.get_hf_token()
    
    api.create_repo(
        repo_id='rouskinlab/' + name,
        token=hf_token,
        exist_ok=exist_ok,
        private=True,
        repo_type="dataset",
    )
    
    api.upload_file(
        path_or_fileobj=datapath,
        path_in_repo='data.json',
        repo_id='rouskinlab/' + name,
        repo_type="dataset",
        token=hf_token,
        commit_message=commit_message,
        **kwargs,
    )

    if add_card:
        card = write_card(datapath)
        api.upload_file(
            path_or_fileobj=card,
            repo_id='rouskinlab/' + name,
            path_in_repo = 'README.md',
            repo_type="dataset",
            token=hf_token,
            commit_message=commit_message,
        )
    
    
def write_card(datapath):
    source = os.path.basename(datapath)
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    name = name_from_path(datapath)

    out = """
---
license: mit
language:
  - en
tags:
  - chemistry
  - biology`
author: Silvi Rouskin
source: {}
date: {}
---


# Data types
""".format(source, date)

    data_type_count = {}
    data = json.load(open(datapath, 'r'))
    for dp in data.values():
        for k, v in dp.items():
            if v == None or (type(v) == float and np.isnan(v)):
                continue
            if k not in data_type_count:
                data_type_count[k] = 0
            data_type_count[k] += 1
    
    for k, v in data_type_count.items():
        out += f"""
- **{k}**: {v} datapoints"""

    #TODO add filtering report
    
    
    # dump
    path = Path(name=name, root=dirname(dirname(datapath)))
    os.makedirs(os.path.dirname(path.get_card()), exist_ok=True)
    with open(path.get_card(), 'w') as f:
        f.write(out)
    return path.get_card()
