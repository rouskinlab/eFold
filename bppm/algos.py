from abc import ABC, abstractmethod
import sys, os
sys.path.append("/Users/yvesmartin/src/efold")
from efold.core.postprocess import HungarianAlgorithm as efoldHungarianAlgorithm
from efold.core.postprocess import postprocess_new_nc
import torch
from util import *
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm

cm = {'black': 'True Negatives', 'white': 'True Positive', 'red': 'False Positives', 'blue': 'False Negatives'}
cmap = ListedColormap(cm.keys())
bounds = np.arange(len(cm) + 1) - 0.5
norm = BoundaryNorm(bounds, cmap.N)
from arnie_utils import *
from scipy.optimize import linear_sum_assignment


class AlgoTemplate(ABC):
    def __init__(self, pred=None, L=None, label=None, bppm=None):
        self.pred = pred
        self.L = L
        self.label = label
        self.bppm = bppm
        
    @property
    def f1(self):
        if self.pred is None:
            raise ValueError("No prediction available")
        if self.label is None:
            raise ValueError("No label available")
        return self.compute_f1()
    
    @f1.setter
    def f1(self, value):
        raise ValueError("f1 is read-only")
    
    @f1.getter  
    def f1(self):
        return self.compute_f1()
    
    @property
    def precision(self):
        if self.pred is None:
            raise ValueError("No prediction available")
        if self.label is None:
            raise ValueError("No label available")
        return self.compute_precision()
    
    @precision.getter
    def precision(self):
        return self.compute_precision()
    
    @property
    def recall(self):
        if self.pred is None:
            raise ValueError("No prediction available")
        if self.label is None:
            raise ValueError("No label available")
        return self.compute_recall()
    
    @recall.getter
    def recall(self):
        return self.compute_recall()
    
    def compute_precision(self, label=None):
        if label is None:
            if self.label is None:
                raise ValueError("No label provided")
            label = self.label
        else:
            self.label = label
        confusion_matrix = self.compute_confusion_matrix(label)
        true_positives = confusion_matrix == 1
        false_positives = confusion_matrix == 3
        return true_positives.sum() / (true_positives.sum() + false_positives.sum())
    
    def compute_recall(self, label=None):
        if label is None:
            if self.label is None:
                raise ValueError("No label provided")
            label = self.label
        else:
            self.label = label
        confusion_matrix = self.compute_confusion_matrix(label)
        true_positives = confusion_matrix == 1
        false_negatives = confusion_matrix == 2
        return true_positives.sum() / (true_positives.sum() + false_negatives.sum())
    
    @abstractmethod
    def run(self, bppm, *args, **kwargs):
        pass
    
    def preprocess(self, bppm):
        return bppm
    
    def compute_f1(self, label=None):
        if label is None:
            if self.label is None:
                raise ValueError("No label provided")
            label = self.label
        else:
            self.label = label
        return f1_score(label.flatten(), self.pred.flatten())
    
    def compute_confusion_matrix(self, label=None, pred=None):
        if label is None:
            if self.label is None:
                raise ValueError("No label provided")
            label = self.label
        if pred is None:
            if self.pred is None:
                raise ValueError("No prediction provided")
            pred = self.pred
        true_negatives = (1 - label) * (1 - pred)
        true_positives = label * pred
        false_positives = (1 - label) * pred
        false_negatives = label * (1 - pred)
        confusion_matrix = true_positives + false_positives * 2 + false_negatives * 3
        assert ((true_negatives == 1) == (confusion_matrix == 0)).all(), "True negatives are not correctly computed"
        return confusion_matrix

    def plot_confusion_matrix(self, label=None, pred=None, ax=None):
        confusion_matrix = self.compute_confusion_matrix(label, pred)
        fig = plt.subplot() if ax is None else ax
        plt.imshow(confusion_matrix, cmap=cmap, norm=norm)
        plt.title("{}, f1={:.2f}".format(self.name, self.compute_f1(label)))
        plt.colorbar(ticks=np.arange(len(cm)), format=lambda x, pos: list(cm.values())[int(x)], shrink=0.8)
        return fig
    
    def plot_bppm(self, bppm=None, ax=None):
        if bppm is None:
            if self.bppm is None:
                raise ValueError("No bppm provided")
            bppm = self.bppm
        fig = plt.subplot() if ax is None else ax
        plt.imshow(bppm, cmap="viridis")
        plt.colorbar(shrink=0.8)
        plt.title("bppm")
        return fig
    
    
class HungarianAlgo(AlgoTemplate):
    name = "Hungarian"
    ha = efoldHungarianAlgorithm()
    
    def run(self, bppm=None, threshold=0.5, **kwargs):
        if bppm is None:
            if self.bppm is None:
                raise ValueError("No bppm provided")
            bppm = self.bppm
        else:
            self.bppm = bppm
        self.pred = self.ha.run(bppm, threshold, **kwargs)[0]
        self.L = bppm.shape[0]
        return self
    
    def preprocess(self, bppm):
        return torch.sigmoid(bppm)


class UFoldAlgo(AlgoTemplate):
    name = "UFold"
    
    def run(self, bppm, sequence):
        if bppm is None:
            if self.bppm is None:
                raise ValueError("No bppm provided")
            bppm = self.bppm
        else:
            self.bppm = bppm
        bppm = self.preprocess(bppm).unsqueeze(0)
        self.pred = np.round(np.array(postprocess_new_nc(
            bppm,
            one_hot_encode(sequence),
            0.01, 0.1, 100, 1.6, True, 1.5))[0])
        self.L = bppm.shape[0]
        return self
    
    def preprocess(self, bppm):
        return bppm
    

class RibonanzaNetAlgo(AlgoTemplate):
    name = "RibonanzaNet"
    
    def run(self, bppm, **kwargs):
        if bppm is None:
            if self.bppm is None:
                raise ValueError("No bppm provided")
            bppm = self.bppm
        else:
            self.bppm = bppm
        bppm = np.array(self.preprocess(bppm))
        self.pred = self._hungarian(bppm, **kwargs)
        self.L = bppm.shape[0]
        return self
    
    def preprocess(self, bppm):
        return bppm
    
    def _sigmoid(self, x, slope_factor=1):
        return 1 / (1 + np.exp(-slope_factor * x))

    def _hungarian(self, bppm, exp=1, sigmoid_slope_factor=None, prob_to_0_threshold_prior=0,
                prob_to_1_threshold_prior=1, theta=0, ln=False, add_p_unpaired=True,
                allowed_buldge_len=0, min_len_helix=2):
        bpp = bppm.copy()
        bpp_orig = bpp.copy()

        if add_p_unpaired:
            p_unpaired = 1 - np.sum(bpp, axis=0)
            for i, punp in enumerate(p_unpaired):
                bpp[i, i] = punp

        # apply prob_to_0 threshold and prob_to_1 threshold
        bpp = np.where(bpp < prob_to_0_threshold_prior, 0, bpp)
        bpp = np.where(bpp > prob_to_1_threshold_prior, 1, bpp)

        # aply exponential. On second thought this is likely not as helpful as sigmoid since
        # * for 0 < exp < 1 lower probs will increase more than higher ones (seems undesirable)
        # * for exp > 1 all probs will decrease, which seems undesirable (but at least lower probs decrease more than higher ones)
        bpp = np.power(bpp, exp)

        # # apply log which follows botlzamann where -ln(P) porportional to Energy
        if ln:
            bpp = np.log(bpp)

        bpp = np.where(np.isneginf(bpp), -1e10, bpp)
        bpp = np.where(np.isposinf(bpp), 1e10, bpp)

        # apply sigmoid modified by slope factor
        if sigmoid_slope_factor is not None and np.any(bpp):
            bpp = self._sigmoid(bpp, slope_factor=sigmoid_slope_factor)

            # should think about order of above functions and possibly normalize again here

            # run hungarian algorithm to find base pairs
        _, row_pairs = linear_sum_assignment(-bpp)
        bp_list = []
        for col, row in enumerate(row_pairs):
            # if bpp_orig[col, row] != bpp[col, row]:
            #    print(col, row, bpp_orig[col, row], bpp[col, row])
            if bpp_orig[col, row] > theta and col < row:
                bp_list.append([col, row])

        structure = convert_bp_list_to_dotbracket(bp_list, bpp.shape[0])
        structure = post_process_struct(structure, allowed_buldge_len, min_len_helix)
        bp_list = convert_dotbracket_to_bp_list(structure, allow_pseudoknots=True)
        matrix = bp_to_matrix(bp_list, bpp.shape[0])
        return matrix

