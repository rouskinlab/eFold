import os

def _source_env(path):
    """
    Source the environment variables from the file at path.

    Args:
        path (str): The path to the file to source.
    """
    out = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.split('#')[0].strip()
            line = line.replace('export', '')
            key, value = line.split('=')
            key, value = key.replace(' ','').replace('"','').strip(), value.replace(' ','').replace('"','').strip()
            os.environ[key] = value
            out[key] = value

    return out


class Environment:
    
    def __init__(self, **kwargs):
        self.HUGGINGFACE_TOKEN = kwargs.get('HUGGINGFACE_TOKEN', os.environ.get('HUGGINGFACE_TOKEN', None)) 
        self.DATA_FOLDER = kwargs.get('DATA_FOLDER', os.environ.get('DATA_FOLDER', 'data/input_files'))
        self.DATA_FOLDER_TESTING = kwargs.get('DATA_FOLDER_TESTING', os.environ.get('DATA_FOLDER_TESTING', 'data/input_files_testing'))
        self.RNASTRUCTURE_PATH = kwargs.get('RNASTRUCTURE_PATH', os.environ.get('RNASTRUCTURE_PATH', ''))
        self.RNASTRUCTURE_TEMP_FOLDER = kwargs.get('RNASTRUCTURE_TEMP_FOLDER', os.environ.get('RNASTRUCTURE_TEMP_FOLDER', 'temp'))
        
    @classmethod
    def from_env(cls, path):
        return cls(**_source_env(path))
    
env = Environment()

def setup_env(path=None, **kwargs):
    """Setup the environment variables.
    
    Args:
        path (str): The path to the file to source.
        **kwargs: The environment variables to set.
            - HUGGINGFACE_TOKEN (str): The HuggingFace token.
            - DATA_FOLDER (str): The path to the data folder.
            - DATA_FOLDER_TESTING (str): The path to the data folder for testing.
            - RNASTRUCTURE_PATH (str): The path to the RNAstructure executable.
            - RNASTRUCTURE_TEMP_FOLDER (str): The path to the temporary folder for RNAstructure.

    Raises:
        ValueError: If an environment variable does not exist.
        
    Examples:
        >>> setup_env('env')
        >>> os.environ['RNASTRUCTURE_TEMP_FOLDER'].split('/')[-1]
        'temp'
    """
    
    # Source the environment variables from the file at path
    global env
    if path:
        env = Environment.from_env(path)

    # Update the environment variables with the kwargs
    for key, value in kwargs.items():
        if hasattr(env, key):
            setattr(env, key, value)
        else:
            raise ValueError(f'Environment variable {key} does not exist.')