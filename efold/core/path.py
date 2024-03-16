from os.path import join
import os
from rouskinhf.env import Env
import numpy as np
import pickle
from rouskinhf.path import Path as RouskinPath


def dont_dump_none(func):
    def wrapper(self, val):
        if val is None:
            return None
        return func(self, val)

    return wrapper


class Path(RouskinPath):
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

    def __init__(self, name, root="data") -> None:
        super().__init__(name=name, root=root)

    def get_data_pickle(self) -> str:
        """Returns the path to the data.pickle file."""
        return join(self.get_main_folder(), "data.pkl")

    def get_reference(self) -> str:
        """Returns the path to the references.txt file."""
        return join(self.get_main_folder(), "references.npy")

    def load_reference(self):
        """Returns the list of references."""
        return np.load(self.get_reference(), allow_pickle=True)

    def dump_reference(self, references):
        """Dumps the list of references."""
        np.save(self.get_reference(), references)

    def get_sequence(self) -> str:
        """Returns the path to the sequences.txt file."""
        return join(self.get_main_folder(), "sequences.npy")

    def load_sequence(self):
        """Returns the list of sequences."""
        return np.load(self.get_sequence(), allow_pickle=True)

    def dump_sequence(self, sequences):
        """Dumps the list of sequences."""
        np.save(self.get_sequence(), sequences)

    def get_length(self) -> str:
        """Returns the path to the lengths.txt file."""
        return join(self.get_main_folder(), "lengths.npy")

    def load_length(self):
        """Returns the list of lengths."""
        return np.load(self.get_length(), allow_pickle=True)

    def dump_length(self, lengths):
        """Dumps the list of lengths."""
        np.save(self.get_length(), lengths)

    def get_dms(self) -> str:
        """Returns the path to the dms.txt file."""
        return join(self.get_main_folder(), "dms.pkl")

    def load_dms(self):
        """Returns the list of dms."""
        if not os.path.exists(self.get_dms()):
            return None
        return pickle.load(open(self.get_dms(), "rb"))

    @dont_dump_none
    def dump_dms(self, val) -> str:
        """Dumps the list of dms."""
        pickle.dump(val, open(self.get_dms(), "wb"))

    def get_shape(self) -> str:
        """Returns the path to the shapes.txt file."""
        return join(self.get_main_folder(), "shapes.pkl")

    def load_shape(self):
        """Returns the list of shapes."""
        if not os.path.exists(self.get_shape()):
            return None
        return pickle.load(open(self.get_shape(), "rb"))

    @dont_dump_none
    def dump_shape(self, val):
        """Dumps the list of shapes."""
        pickle.dump(val, open(self.get_shape(), "wb"))

    def get_structure(self) -> str:
        """Returns the path to the structures.txt file."""
        return join(self.get_main_folder(), "structures.pkl")

    def load_structure(self):
        """Returns the list of structures."""
        if not os.path.exists(self.get_structure()):
            return None
        return pickle.load(open(self.get_structure(), "rb"))

    @dont_dump_none
    def dump_structure(self, val):
        """Dumps the list of structures."""
        pickle.dump(val, open(self.get_structure(), "wb"))
