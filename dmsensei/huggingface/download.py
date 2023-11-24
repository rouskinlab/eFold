from .path import Path
from huggingface_hub import snapshot_download
from .env import Env

def download_data(name:str):
    path = Path(name=name)
    snapshot_download(
            repo_id='rouskinlab/' + name,
            repo_type="dataset",
            local_dir=path.get_main_folder(),
            token=Env.get_hf_token(),
            allow_patterns=["data.json"],
        )
    