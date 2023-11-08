import pandas as pd
import numpy as np
import pickle

def format_to_ribonanza(prediction):

    prediction = np.vstack(prediction).reshape(-1, 2)    
    prediction = np.clip(prediction, 0, 1)

    prediction = pd.DataFrame(prediction, columns=["reactivity_DMS_MaP", "reactivity_2A3_MaP"]).reset_index().rename(columns={"index": "id"})

    return prediction