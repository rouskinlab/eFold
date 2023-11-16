from ..config import REFERENCE_METRIC
import numpy as np
from ..core.datapoint import Datapoint
class Stack:
    def __init__(self, L, mode, data_type):
        self.L = L
        self.vals = []
        assert mode in ["best", "worse"], "mode must be in ['best','worse']"
        self.mode = mode
        self.data_type = data_type

    def _sort(self):
        self.vals.sort(
            key=lambda dp: self._get_scores(dp) if self._get_scores(dp) is not None else -np.inf,
            reverse=self.mode == "best",
        )
        
    def _get_scores(self, dp:Datapoint):
        return dp.read_reference_metric(self.data_type)
        

    def _should_replace(self, current_score, new_score):
        if self.mode == "best":
            return current_score < new_score
        return current_score > new_score

    def try_to_add(self, dp):
        if len(self.vals) < self.L:
            self.vals.append(dp)
            self._sort()
            return True

        elif self._should_replace(
            self._get_scores(self.vals[-1]),
            self._get_scores(dp),
        ):
            self.vals.pop()
            self.vals.append(dp)
            self._sort()
            return True
        return False

    def is_empty(self):
        return not len(self.vals)
