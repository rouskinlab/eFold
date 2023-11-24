from .path import Path
from huggingface_hub import HfApi
import os
import datetime
import json
from .env import Env

def upload_dataset(datapath:dict, exist_ok= False, commit_message:str=None, add_card=True, **kwargs):
    api = HfApi()
    name = datapath.split('/')[-2]

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
            **card,
        )
    
    
def write_card(datapath):
    source = os.path.basename(datapath)
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
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
        for k in dp.keys():
            if k not in data_type_count:
                data_type_count[k] = 0
            data_type_count[k] += 1
    
    for k, v in data_type_count.items():
        out += f"""
## {k}
- {v} datapoints
"""

    #TODO add filtering report
    return out
