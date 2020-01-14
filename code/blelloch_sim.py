import copy
import numpy as np
import random
import torch


def ceil_log(n):
    return int(np.ceil(np.log2(n)))


def floor_log(n):
    return int(np.floor(np.log2(n)))


def blelloch(a, op, I, L_m=float('inf')):
    """ A simulation of a *heterogeneous* Blelloch scan algorithm. The
    difference between this and a normal Blelloch scan algorithm is that you
    can control the number of levels that up-sweep and down-sweep performs by
    setting L_m.
    """
    print(a, op, I, L_m)

    if isinstance(a, np.ndarray):
        n = a.shape[0] - 1
    else:
        n = len(a) - 1
    d = -1
    for d in range(0, min(ceil_log(n + 1) - 2 + 1, L_m)):
        for i in range(0, n - pow(2, d) + 1, pow(2, d + 1)):
            l, r = i + pow(2, d) - 1, min(i + pow(2, d + 1) - 1, n)
            a[r] = op(a[l], a[r])

    #a[n] = I
    reduction = I
    d += 1
    for i in range(0, n - pow(2, d) + 1, pow(2, d)):
        idx = i + pow(2, d) - 1
        T = reduction
        reduction = op(reduction, a[idx])
        a[idx] = T
    a[n] = reduction

    for d in reversed(range(0, min(ceil_log(n + 1) - 2 + 1, L_m))):
        for i in range(0, n - pow(2, d) + 1, pow(2, d + 1)):
            l, r = i + pow(2, d) - 1, min(i + pow(2, d + 1) - 1, n)
            T = a[l]
            a[l] = a[r]
            a[r] = op(a[r], T)


def linear(a, op, I):
    if isinstance(a, np.ndarray):
        n = a.shape[0]
    else:
        n = len(a)
    s = I
    for i in range(n):
        s_prev = s
        s = op(s, a[i])
        a[i] = s_prev


def gen_input(chain_len=100,
              matrix_max_size=20,
              matrix_size=None,
              dtype=np.int,
              low=0,
              high=1,
              batch_size=None,
              to_torch=False,
              reverse=False):
    h = (np.random.randint(1, matrix_max_size)
         if matrix_size is None else matrix_size)
    w = (np.random.randint(1, matrix_max_size)
         if matrix_size is None else matrix_size)
    res = []
    for _ in range(chain_len):
        if batch_size is None:
            size = (h, w)
        else:
            size = (batch_size, h, w)
        if np.issubdtype(dtype, np.integer):
            new_matrix = np.random.randint(low, high, size=size, dtype=dtype)
        else:
            new_matrix = (np.random.rand(*size).astype(dtype) * (high - low) +
                          low)
        if to_torch:
            new_matrix = torch.from_numpy(new_matrix)
        res.append(new_matrix)
        h = w
        w = (np.random.randint(1, matrix_max_size)
             if matrix_size is None else matrix_size)
    if reverse:
        return list(reversed(res))
    return res


class Identity(object):

    def __init__(self):
        self.shape = 'I'


def test():
    chain_len = random.randint(0, 999)
    expect_a = gen_input(
        chain_len=chain_len,
        dtype=np.float,
        low=-1.0,
        high=1.0,
        reverse=True,
    )
    actual_a = copy.deepcopy(expect_a)

    def matmul(a, b):
        if isinstance(a, Identity):
            return b
        elif isinstance(b, Identity):
            return a
        else:
            return b @ a

    blelloch(
        actual_a,
        matmul,
        Identity(),
        L_m=random.randint(
            0,
            ceil_log(chain_len) * 2,
        ),
    )
    linear(expect_a, matmul, Identity())
    for i in range(1, len(actual_a)):
        np.testing.assert_allclose(actual_a[i], expect_a[i], rtol=1e-04)


def main():
    for _ in range(100):
        test()


if __name__ == '__main__':
    main()
