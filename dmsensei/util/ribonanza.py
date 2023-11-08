import pandas as pd
import numpy as np

def format_to_ribonanza(trainer_prediction):

    out = np.concatenate(trainer_prediction, axis=0).reshape(-1, 2)
    out = np.clip(out, 0, 1)

    out = pd.DataFrame(out, columns=["reactivity_DMS_MaP", "reactivity_2A3_MaP"]).reset_index().rename(columns={"index": "id"})

    return out