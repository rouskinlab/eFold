from os.path import join
import os
from .env import Env

class Path:
    """Path to files and folders of a datafolder of name `name`. The path to the data folder is `DATA_FOLDER`, which is defined in `env`.

    Parameters
    ----------

    name : str

    Returns
    -------

    PathDatafolder

    Example
    -------

    >>> path = PathDatafolder(name='my_test_datafolder_pytest')
    >>> path.name
    'my_test_datafolder_pytest'
    >>> print(path)
    PathDatafolder(name='my_test_datafolder_pytest')
    """

    def __init__(self, name, root = Env.get_data_folder()) -> None:
        assert type(name) == str, f'name {name} is not a string'
        assert type(root) == str, f'root {root} is not a string'
        self.root = root
        self.name = name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    def get_data_folder(self)->str:
        """Returns the path to the data folder."""
        return self.root

    def get_main_folder(self)->str:
        """Returns the path to the main folder."""
        return join(self.get_data_folder(), self.name)
    
    def get_data_json(self)->str:
        """Returns the path to the data.json file."""
        return join(self.get_main_folder(), 'data.json')
    
    def get_data_pickle(self)->str:
        """Returns the path to the data.pickle file."""
        return join(self.get_main_folder(), 'data.pkl')
    
    def get_card(self)->str:
        """Returns the path to the README.md file."""
        return join(self.get_main_folder(), 'README.md')