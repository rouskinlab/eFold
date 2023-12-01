import os


class Env:
    def get_hf_token() -> str:
        if "HUGGINGFACE_TOKEN" in os.environ:
            return os.environ["HUGGINGFACE_TOKEN"]
        raise Exception("HUGGINGFACE_TOKEN not found in environment variables")

    def get_data_folder() -> str:
        if "DATA_FOLDER" in os.environ:
            return os.environ["DATA_FOLDER"]
        return "data/datafolders"
