import torch
import math
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from ..config import seq2int

class HungarianAlgorithm:
        
    def run(self, bppm, threshold=0.5, canonical_bp_only=False, sequences=None, min_hairpin_loop=3, output_format='matrix'):
        """Runs the Hungarian algorithm on the input bppm matrix
        
        Args:
        - bppm (torch.Tensor): batch_size x n x n matrix of base pair probabilities
        - threshold (float): the minimum threshold for a base pair to be considered as a pair candidate
        - canonical_bp_only (bool): if True, only AU, CG, GU and pairs with N are considered as pair candidates
        - sequences (list): list of RNA sequences (only used if canonical_bp_only is True)
        - min_hairpin_loop (int): the minimum number of bases in a hairpin loop in the output structure
        - output_format (str): the output format of the base pairs (either 'matrix' or 'list')
        
        Example:
        >>> inpt = np.diag(np.ones(10))[::-1]
        >>> inpt += np.random.normal(0, 0.2, inpt.shape)
        >>> inpt = (inpt + inpt.T)/2 
        >>> inpt = torch.tensor(inpt).unsqueeze(0)
        >>> seq = np.array([seq2int[a] for a in 'GAACUAUUCU'])
        >>> out = HungarianAlgorithm().run(inpt, threshold=0.5, canonical_bp_only=True, sequences=[seq]).squeeze(0).tolist()
        >>> assert out == [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],\
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],\
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],\
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],\
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],\
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],\
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],\
                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],\
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],\
                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], "The output is not as expected: {}".format(out)
        >>> out = HungarianAlgorithm().run(torch.tensor([[[0., 0.6, 0.8],[0.6, 0., 0.9],[0.8, 0.9, 0.]]]), threshold=0.5, min_hairpin_loop=0).squeeze(0).tolist()
        >>> assert out == [[0., 1., 0.],[0., 0., 1.],[1., 0., 0.]], "The output is not as expected: {}".format(out)
        """
        
        # make sure that the input dimension is batch_size x n x n
        if len(bppm.shape) == 2:
            bppm = bppm.unsqueeze(0)
        assert len(bppm.shape) == 3, "The input bppm matrix should be batch_size x n x n"
        assert bppm.shape[1] == bppm.shape[2], "The input bppm matrix should be batch_size x n x n"
        
        # just work with numpy (needed for the optimization step)
        bppm = bppm.cpu().numpy()
        
        # run hungarian algorithm for each batch
        if output_format == 'matrix':
            bp_matrix = np.zeros(bppm.shape) 
        elif output_format == 'list':
            bp_list = []
        else:
            raise ValueError("The output format should be either 'matrix' or 'list'")
        
        for i in range(bppm.shape[0]):
            if sequences is not None:
                assert type(sequences[i]) == np.ndarray, "The input sequences should be a numpy array"
            assert self.is_symmetric(bppm[i]), "The input bppm matrix should be symmetric"
            
            # remove non-canonical base pairs and too short hairpins
            if canonical_bp_only:
                bppm[i] = self._remove_non_canonical(bppm[i], sequences[i])
            bppm[i] = self._remove_short_loop(bppm[i], min_hairpin_loop)
            
            # run hungarian algorithm only on rows and columns that have at least one value greater than threshold
            compression_idx = self._pairable_bases(bppm[i], threshold)
            compressed_bppm = bppm[i][compression_idx][:, compression_idx]
            row_ind, col_ind = self._hungarian_algorithm(compressed_bppm)         
            
            # convert the result to the original sized matrix
            for compressed_row, compressed_col in zip(row_ind, col_ind):
                bppm_row, bppm_col = compression_idx[compressed_row], compression_idx[compressed_col]
                if bppm[i, bppm_row, bppm_col] > threshold:
                    if output_format == 'matrix':
                        bp_matrix[i, bppm_row, bppm_col] = 1
                        bp_matrix[i, bppm_col, bppm_row] = 1
                    elif output_format == 'list':
                        bp_list.append((bppm_row, bppm_col))
        return bp_matrix if output_format == 'matrix' else bp_list
    
    def _hungarian_algorithm(self, cost_matrix):
        """Returns the row and column indices of the optimal assignment using the Hungarian algorithm"""
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        return row_ind, col_ind
    
    def _pairable_bases(self, bppm, threshold):
        """Returns the indices of rows that have at least one value greater than threshold
        
        Example:
        >>> assert (HungarianAlgorithm()._pairable_bases(np.array([[0.0, 0.0, 0.0], [0.0, 0.6, 0.8], [0.0, 0.8, 0.9]]), 0.01) == np.array([1, 2])).all(), "The output is not as expected"
        """
        return np.where((bppm > threshold).any(axis=0))[0]
    
    def is_symmetric(self, bppm):
        return (bppm == bppm.transpose(1, 0)).all()
    
    def _remove_non_canonical(self, bp_matrix, sequence):
        """This function removes non-canonical base pairs (keeps only AU, CG, GU, and all base pairs with N)
        
        Logic: 
            - embed the sequence
            - sum the sequence with its transpose
            - canonical base pairs sum into 5 (AU, CG) and 7 (GU)
            
            Mapping: A=1, C=2, G=3, U=4
            Which gives, in the sum:    
                A | C | G | U 
            A | 2 | 3 | 4 | 5
            C | 3 | 4 | 5 | 6
            G | 4 | 5 | 6 | 7
            U | 5 | 6 | 7 | 8
            
            -> if the sum is 5 or 7, keep the base pair
        
        Example:
        >>> seq = np.array([seq2int[a] for a in 'ACGU'])
        >>> out = HungarianAlgorithm()._remove_non_canonical(np.ones((4,4)), seq).tolist()
        >>> assert (out == np.array([[0., 0., 0., 1.],[0.,0.,1.,0.],[0.,1.,0.,1.],[1.,0.,1.,0.]])).all(), "The output is not as expected: {}".format(out)
        """
        # make the pairing matrix
        sequence = sequence.reshape(-1, 1)
        mat = sequence + sequence.T
        
        # make sure that this function matches the embedding from config
        assert set([seq2int[a] + seq2int[b] for a, b in (("A","U"), ("C","G"), ("G","U"))]) == set([5, 7]), "The embedding does not match the expected values"
        
        # remove non-canonical base pairs
        bp_matrix[(mat!=5) & (mat!=7)] = 0
        
        return bp_matrix
    
    def _remove_short_loop(self, bp_matrix, min_hairpin_loop):
        """Removes hairpins that are too short and self-pairing
        
        Example:
        >>> (HungarianAlgorithm()._remove_short_loop(np.ones((5,5)), 2) == [[0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0]]).all()
        True
        """
        n = bp_matrix.shape[1]
        for i in range(n):
            bp_matrix[i, max(0, i-min_hairpin_loop):min(n, i+min_hairpin_loop+1)] = 0
        return bp_matrix
    

def constraint_matrix_batch(x):
    """
    this function is referred from e2efold utility function, located at https://github.com/ml4bio/e2efold/tree/master/e2efold/common/utils.py
    """
    base_a = x[:, :, 0]
    base_u = x[:, :, 1]
    base_c = x[:, :, 2]
    base_g = x[:, :, 3]
    batch = base_a.shape[0]
    length = base_a.shape[1]
    au = torch.matmul(base_a.view(batch, length, 1), base_u.view(batch, 1, length))
    au_ua = au + torch.transpose(au, -1, -2)
    cg = torch.matmul(base_c.view(batch, length, 1), base_g.view(batch, 1, length))
    cg_gc = cg + torch.transpose(cg, -1, -2)
    ug = torch.matmul(base_u.view(batch, length, 1), base_g.view(batch, 1, length))
    ug_gu = ug + torch.transpose(ug, -1, -2)
    return au_ua + cg_gc + ug_gu

def constraint_matrix_batch_addnc(x):
    x = x.type(torch.float)
    base_a = x[:, :, 0]
    base_u = x[:, :, 1] #3
    base_c = x[:, :, 2] #1
    base_g = x[:, :, 3] #2
    batch = base_a.shape[0]
    length = base_a.shape[1]
    au = torch.matmul(base_a.view(batch, length, 1), base_u.view(batch, 1, length))
    au_ua = au + torch.transpose(au, -1, -2)
    cg = torch.matmul(base_c.view(batch, length, 1), base_g.view(batch, 1, length))
    cg_gc = cg + torch.transpose(cg, -1, -2)
    ug = torch.matmul(base_u.view(batch, length, 1), base_g.view(batch, 1, length))
    ug_gu = ug + torch.transpose(ug, -1, -2)
    ## add non-canonical pairs
    ac = torch.matmul(base_a.view(batch, length, 1), base_c.view(batch, 1, length))
    ac_ca = ac + torch.transpose(ac, -1, -2)
    ag = torch.matmul(base_a.view(batch, length, 1), base_g.view(batch, 1, length))
    ag_ga = ag + torch.transpose(ag, -1, -2)
    uc = torch.matmul(base_u.view(batch, length, 1), base_c.view(batch, 1, length))
    uc_cu = uc + torch.transpose(uc, -1, -2)
    aa = torch.matmul(base_a.view(batch, length, 1), base_a.view(batch, 1, length))
    uu = torch.matmul(base_u.view(batch, length, 1), base_u.view(batch, 1, length))
    cc = torch.matmul(base_c.view(batch, length, 1), base_c.view(batch, 1, length))
    gg = torch.matmul(base_g.view(batch, length, 1), base_g.view(batch, 1, length))
    return au_ua + cg_gc + ug_gu + ac_ca + ag_ga + uc_cu + aa + uu + cc + gg

def contact_a(a_hat, m):
    a = a_hat * a_hat
    a = (a + torch.transpose(a, -1, -2)) / 2
    a = a * m
    return a

def sign(x):
    return (x > 0).type(x.dtype)


def soft_sign(x):
    k = 1
    return 1.0/(1.0+torch.exp(-2*k*x))


def postprocess_new(u, x, lr_min, lr_max, num_itr, rho=0.0, with_l1=False,s=math.log(9.0)):
    """
    :param u: utility matrix, u is assumed to be symmetric, in batch
    :param x: RNA sequence, in batch
    :param lr_min: learning rate for minimization step
    :param lr_max: learning rate for maximization step (for lagrangian multiplier)
    :param num_itr: number of iterations
    :param rho: sparsity coefficient
    :param with_l1:
    :return:
    """
    m = constraint_matrix_batch(x).float()
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

def postprocess_new_nc(u, x, lr_min, lr_max, num_itr, rho=0.0, with_l1=False,s=math.log(9.0)):
    """
    :param u: utility matrix, u is assumed to be symmetric, in batch
    :param x: RNA sequence, in batch
    :param lr_min: learning rate for minimization step
    :param lr_max: learning rate for maximization step (for lagrangian multiplier)
    :param num_itr: number of iterations
    :param rho: sparsity coefficient
    :param with_l1:
    :return:
    """
    m = constraint_matrix_batch_addnc(x).float()
    #m = 1.0
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
