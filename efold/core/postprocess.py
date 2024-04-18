import torch
import math
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from ..config import seq2int

class Constraints:

    """ Compute a mask of constraints to remove sharp loops and non-canonical pairs from the input matrix 

    Args:
    - input_matrix (torch.Tensor): n x n matrix of base pair probabilities
    - min_hairpin_length (int): minimum length of hairpin loops
    - canonical_only (bool): if True, only allow A-U, G-C, and G-U pairs
    - sequence (torch.Tensor or string): sequence of bases

    Example:
    >>> inpt = torch.tensor([[0.1, 0.6, 0.8],[0.6, 0.1, 0.9],[0.8, 0.9, 0.1]])
    >>> sequence = torch.tensor([seq2int[a] for a in "GCU"])
    >>> out = Constraints().apply_constraints(inpt, sequence=sequence, min_hairpin_length=0, canonical_only=True)
    >>> assert (out == torch.tensor([[0.0, 0.6, 0.8],[0.6, 0.0, 0.0],[0.8, 0.0, 0.0]])).all(), "The output is not as expected: {}".format(out)

    """

    def apply_constraints(self, input_matrix, min_hairpin_length=3, canonical_only=True, sequence=None):

        # mask elements of the diagonals and sub-diagonals
        mask = self.mask_sharpLoops(input_matrix, min_hairpin_length)

        # Mask elements of the matrix that are not A-U, G-C, or G-U pairs using the sequence
        if canonical_only: mask *= self.mask_nonCanonical(sequence)
        
        return input_matrix * mask

    def mask_sharpLoops(self, input_matrix, min_hairpin_length):

        # mask elements of the diagonals and sub-diagonals
        mask = np.tri(input_matrix.shape[0], k=-min_hairpin_length-1).astype(int)
        return torch.tensor(mask + mask.T)

    def mask_nonCanonical(self, sequence):

        # Embed sequence
        if type(sequence) == str: sequence = torch.tensor([seq2int[a] for a in sequence])
        
        # make the pairing matrix
        sequence = sequence.reshape(-1, 1)
        pair_of_bases = sequence + sequence.T

        # find the allowable pairs
        allowable_pair = set()
        for pair in ["GU", "GC", "AU"]: allowable_pair.add(seq2int[pair[0]] + seq2int[pair[1]])
        allowable_pair = torch.tensor(list(allowable_pair))

        return torch.isin(pair_of_bases, allowable_pair).int()



class HungarianAlgorithm:
        
    def run(self, bppm, threshold=0.5):
        """Runs the Hungarian algorithm on the input bppm matrix
        
        Args:
        - bppm (torch.Tensor): n x n matrix of base pair probabilities
        
        Example:
        >>> inpt = np.diag(np.ones(10))[::-1]
        >>> inpt += np.random.normal(0, 0.2, inpt.shape)
        >>> inpt = (inpt + inpt.T)/2 
        >>> inpt = torch.tensor(inpt)
        >>> out = HungarianAlgorithm().run(inpt)
        >>> assert (out == [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],\
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],\
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],\
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],\
                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],\
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],\
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],\
                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],\
                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],\
                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).all(), "The output is not as expected: {}".format(out)
        >>> out = HungarianAlgorithm().run(torch.tensor([[0., 0.6, 0.8],[0.6, 0., 0.9],[0.8, 0.9, 0.]]) )
        >>> assert (out == [[0., 1., 1.],[1., 0., 1.],[1., 1., 0.]]).all(), "The output is not as expected: {}".format(out)
        """
        
        assert len(bppm.shape) == 2, "The input bppm matrix should be n x n"
        assert bppm.shape[0] == bppm.shape[1], "The input bppm matrix should be n x n"

        assert self.is_symmetric(bppm), "The input bppm matrix should be symmetric"
        
        # just work with numpy (needed for the optimization step)
        if type(bppm)==torch.Tensor: bppm = bppm.cpu().numpy()
        
        # run hungarian algorithm 
        bp_matrix = np.zeros(bppm.shape) 
        
        # run hungarian algorithm only on rows and columns that have at least one value greater than threshold
        compression_idx = self._pairable_bases(bppm, threshold)
        compressed_bppm = bppm[compression_idx][:, compression_idx]
        row_ind, col_ind = self._hungarian_algorithm(compressed_bppm)         
        
        # convert the result to the original sized matrix
        for compressed_row, compressed_col in zip(row_ind, col_ind):
            bppm_row, bppm_col = compression_idx[compressed_row], compression_idx[compressed_col]
            if bppm[bppm_row, bppm_col] > threshold:
                bp_matrix[bppm_row, bppm_col] = 1
                bp_matrix[bppm_col, bppm_row] = 1

        return torch.tensor(bp_matrix)
    
    def _hungarian_algorithm(self, cost_matrix):
        """Returns the row and column indices of the optimal assignment using the Hungarian algorithm"""
        row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
        return row_ind, col_ind
    
    def _pairable_bases(self, bppm, threshold):
        """Returns the indices of rows that have at least one value greater than threshold
        
        Example:
        >>> assert (HungarianAlgorithm()._pairable_bases(np.array([[0.0, 0.0, 0.0], [0.0, 0.6, 0.8], [0.0, 0.8, 0.9]]), 0.01) == np.array([1, 2])).all(), "The output is not as expected"
        """
        return np.where((bppm > threshold).any(axis=0))[0]
    
    def is_symmetric(self, bppm):
        return (bppm == bppm.transpose(1, 0)).all()
    


class UFold_processing:

    def run(self, bppm):
        return self.postprocess(u=bppm)
        
    def postprocess(self, u, lr_min=0.01, lr_max=0.1, num_itr=100, rho=1.6, with_l1=True,s=1.5):
        """
        :param u: utility matrix, u is assumed to be symmetric
        :param lr_min: learning rate for minimization step
        :param lr_max: learning rate for maximization step (for lagrangian multiplier)
        :param num_itr: number of iterations
        :param rho: sparsity coefficient
        :param with_l1:
        :return:
        """
        def soft_sign(x):
            k = 1
            return 1.0/(1.0+torch.exp(-2*k*x))
        
        def contact_a(a_hat, m):
            a = a_hat * a_hat
            a = (a + torch.transpose(a, -1, -2)) / 2
            a = a * m
            return a

        m = 1.0
        # u with threshold
        # equivalent to sigmoid(u) > 0.9
        # u = (u > math.log(9.0)).type(torch.FloatTensor) * u
        u = soft_sign(u - s) * u

        # initialization
        a_hat = (torch.sigmoid(u)) * soft_sign(u - s).detach()
        lmbd = F.relu(torch.sum(contact_a(a_hat, m), dim=-1) - 1).detach()

        # gradient descent
        for t in range(num_itr):

            grad_a = (lmbd * soft_sign(torch.sum(contact_a(a_hat, m), dim=-1) - 1)).unsqueeze_(-1).expand(u.shape) - u / 2
            grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))
            a_hat -= lr_min * grad
            lr_min = lr_min * 0.99

            if with_l1:
                a_hat = F.relu(torch.abs(a_hat) - rho * lr_min)

            lmbd_grad = F.relu(torch.sum(contact_a(a_hat, m), dim=-1) - 1)
            lmbd += lr_max * lmbd_grad
            lr_max = lr_max * 0.99

            # print
            # if t % 20 == 19:
            #     n1 = torch.norm(lmbd_grad)
            #     grad_a = (lmbd * soft_sign(torch.sum(contact_a(a_hat, m), dim=-1) - 1)).unsqueeze_(-1).expand(u.shape) - u / 2
            #     grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))
            #     n2 = torch.norm(grad)
            #     print([t, 'norms', n1, n2, aug_lagrangian(u, m, a_hat, lmbd), torch.sum(contact_a(a_hat, u))])

        a = a_hat * a_hat
        a = (a + torch.transpose(a, -1, -2)) / 2
        a = a * m
        return a
    

class Postprocess:

    def __init__(self, threshold=0.5, canonical_only=False, min_hairpin_length=3):
        self.threshold = threshold
        self.canonical_only = canonical_only
        self.min_hairpin_length = min_hairpin_length

    def run(self, bppms, sequence):

        if len(bppms.shape) == 2:
            bppms = bppms.unsqueeze(0)

        pairing_matrices = []
        for bppm in bppms:

            pairing_matrix = Constraints().apply_constraints(bppm, sequence=sequence,
                                                            min_hairpin_length=self.min_hairpin_length, 
                                                            canonical_only=self.canonical_only)
            
            pairing_matrix = UFold_processing().run(pairing_matrix)

            pairing_matrix = HungarianAlgorithm().run(pairing_matrix, threshold=self.threshold)

            pairing_matrices.append(pairing_matrix)

        return (torch.stack(pairing_matrices) > self.threshold).type(torch.int)

