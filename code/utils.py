import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.sparse as sparse

def num_flat_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def sparsity(m):
    size = m.size()
    num_nonzeros = torch.nonzero(m).size()[0]
    return 1 - num_nonzeros/(size[0] * size[1] * size[2])

def compare_array(a, b, rtol=1e-4):
    try:
        np.testing.assert_allclose(a, b, rtol=rtol)
    except AssertionError as e:
        print(e)

def init_target_params_from_reference(tgt, rfr):
    with torch.no_grad():
        params_tgt = list(tgt.parameters())
        params_rfr = list(rfr.parameters())
        assert len(params_tgt) == len(params_rfr)
        for i in range(len(params_tgt)):
            param_tgt = params_tgt[i]
            param_rfr = params_rfr[i]
            param_tgt.data = param_rfr.data

def save_jcbT_chain(jcbT_chain, path):
    """ Save the chain of Jacobian matrices into path.

    The format of jcbT_chain should be a list. Each element can either be a
    list of csr_matrix (in the case where Jacobian is different for every sample
    in a batch, such as ReLU and MaxPool2d); or it can be a csr_matrix (in the
    case where Jacobian is the same for every sample in a batch, such as Conv2d
    and Linear).

    path points to a directory.
    """
    os.makedirs(path, exist_ok=True)
    for i, jcbT in enumerate(jcbT_chain):
        if isinstance(jcbT, list):
            os.makedirs(os.path.join(path, str(i)), exist_ok=True)
            for j, jcbT_sample in enumerate(jcbT):
                sparse.save_npz(os.path.join(path, str(i), str(j)), jcbT_sample)
        else:
            assert isinstance(jcbT, sparse.csr_matrix)
            sparse.save_npz(os.path.join(path, str(i)), jcbT)

def load_jcbT_chain(path):
    """ The counter-part of save_jcbT_chain. """
    i = 0
    ret = []
    while True:
        path_obj = pathlib.Path(os.path.join(path, '{}.npz'.format(i)))
        if path_obj.is_file():
            ret.append(sparse.load_npz(path_obj))
        else:
            path_obj = pathlib.Path(os.path.join(path, str(i)))
            if path_obj.is_dir():
                jcbT = []
                j = 0
                while True:
                    path_obj = pathlib.Path(
                        os.path.join(path, str(i), '{}.npz'.format(j)))
                    if path_obj.is_file():
                        jcbT.append(sparse.load_npz(path_obj))
                    else:
                        break
                    j += 1
                ret.append(jcbT)
            else:
                break
        i += 1
    return ret
