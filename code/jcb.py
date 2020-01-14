import numpy as np
import scipy.sparse as sparse
import torch
from utils import num_flat_features

def jacobianT(x, y):
    n = x.size()[0]
    dx = num_flat_features(x)
    dy = num_flat_features(y)
    jcbT_rows = []
    for y_dim in range(dy):
        fake_grad_y = torch.zeros(n, dy)
        fake_grad_y[:, y_dim] = 1
        fake_grad_y = fake_grad_y.view(y.size())
        jcbT_per_row = torch.autograd.grad(y, x, grad_outputs=fake_grad_y,
                                           retain_graph=True)[0]
        jcbT_per_row = jcbT_per_row.view(n, dx)
        jcbT_rows.append(jcbT_per_row)
    jcbT = torch.stack(jcbT_rows, dim=2)
    assert jcbT.size() == (n, dx, dy)
    return jcbT

def jacobianT_fake(x, y):
    n = x.size()[0]
    dx = num_flat_features(x)
    dy = num_flat_features(y)
    jcbT = torch.zeros(n, dx, dy)
    return jcbT

def torch_bmtensor_to_lm(bm):
    n = bm.size()[0]  # batch size.
    bm = bm.numpy()
    return [np.squeeze(mat, axis=0) for mat in np.split(bm, n, axis=0)]

def torch_bmtensor_to_lspm(bm):
    return [sparse.csr_matrix(mat) for mat in torch_bmtensor_to_lm(bm)]

def jacobianT_sparse(x, y):
    return torch_bmtensor_to_lspm(jacobianT(x, y))

def sizeof_jacobianT(jcbT):
    if isinstance(jcbT, torch.Tensor):  # Dense
        return jcbT.numpy().data.nbytes
    elif isinstance(jcbT, list) and isinstance(jcbT[0], sparse.csr_matrix):
        # Sparse
        total_size = 0
        for spm in jcbT:
            total_size += spm.data.nbytes
        return total_size
    elif isinstance(jcbT, sparse.csr_matrix):
        return jcbT.data.nbytes
    else:
        raise RuntimeError('jcbT type {} not recognized: {}'.format(type(jcbT),
                                                                    jcbT))
