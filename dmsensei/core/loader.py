from os.path import dirname
from os import makedirs, listdir
import torch
import os

class Loader:
    def __init__(
        self,
        path:str,
    ) -> None:
        self.path = path
        makedirs(dirname(self.get_path()), exist_ok=True)
        
    @classmethod
    def find_best_model(cls, prefix):
        models = [model for model in listdir('models') if model.startswith(prefix)]
        models.sort(key=lambda x: float(x.split('_loss')[-1].split('.')[0].replace('-','.')))
        if len(models) == 0:
            return None
        return cls(path='models/'+models[0])

    def get_path(self):
        return self.path
    
    def get_name(self):
        return self.path.split('/')[-1].split('.')[0]
    
    def write_in_log(self, epoch, loss): 
        with open('models/loss_log.txt', 'a') as f:
            f.write(f'{epoch} {self.get_name()}\t{loss}\n')
        return self

    def load_from_weights(self, safe_load=True):
        if (
            safe_load
            and os.path.exists(self.get_path())
            or not safe_load
        ):
            return torch.load(self.get_path())

    def dump(self, model):
        torch.save(model.state_dict(), self.get_path())
        return self
