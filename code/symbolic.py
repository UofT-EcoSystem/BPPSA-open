import numpy as np
import random
import scipy.sparse as sparse
from numba import jit, prange


@jit(nopython=True, nogil=True, parallel=True)
def _intersect1d(a_indices, a_base, b_indices, b_base):
    """ Assuming a_indices and b_indices are all sorted.
    """
    a_ptr = 0
    b_ptr = 0
    #intersect = []
    flop = 0
    while a_ptr < a_indices.shape[0] and b_ptr < b_indices.shape[0]:
        if a_indices[a_ptr] < b_indices[b_ptr]:
            a_ptr += 1
        elif a_indices[a_ptr] > b_indices[b_ptr]:
            b_ptr += 1
        else:
            #intersect.append((a_ptr + a_base, b_ptr + b_base))
            flop += 2
            a_ptr += 1
            b_ptr += 1
    #return intersect
    return flop


@jit(nopython=True, nogil=True, parallel=True)
def _symbolic_csr_csc_gemm(a_dot_indices,
                           a_dot_indptr,
                           b_dot_indices,
                           b_dot_indptr,
                           c_dot_indices,
                           c_dot_indptr,
                           product_format_csr=True):
    #result = []
    flop_bins = np.zeros(c_dot_indptr.shape[0] - 1)
    for i in prange(c_dot_indptr.shape[0] - 1):
        c_indices = c_dot_indices[c_dot_indptr[i]:c_dot_indptr[i + 1]]
        for j, index in np.ndenumerate(c_indices):
            c_row_idx, c_col_idx = np.int32(0), np.int32(0)
            if product_format_csr:
                c_row_idx = np.int32(i)
                c_col_idx = np.int32(index)
            else:
                c_row_idx = np.int32(index)
                c_col_idx = np.int32(i)
            # Based on c_row_idx and c_col_idx, slice a and b in different ways.
            # a is a csr_matrix
            # This is the indices of one row of a.
            a_indices_base_idx = a_dot_indptr[c_row_idx]
            a_indices = a_dot_indices[a_dot_indptr[c_row_idx]:a_dot_indptr[
                c_row_idx + 1]]
            # This is the indices of one column of b.
            b_indices_base_idx = b_dot_indptr[c_col_idx]
            b_indices = b_dot_indices[b_dot_indptr[c_col_idx]:b_dot_indptr[
                c_col_idx + 1]]
            #result.append(_intersect1d(a_indices, a_indices_base_idx,
            #                           b_indices, b_indices_base_idx))
            flop_bins[i] += _intersect1d(a_indices, a_indices_base_idx,
                                         b_indices, b_indices_base_idx)
    #return result
    return np.sum(flop_bins)


def symbolic_csrgemm(a, b):
    assert isinstance(a, sparse.csr_matrix)
    assert isinstance(b, sparse.csr_matrix)
    c = a @ b
    b = sparse.csc_matrix(b)
    a.sort_indices()
    b.sort_indices()
    c.sort_indices()
    return _symbolic_csr_csc_gemm(a.indices,
                                  a.indptr,
                                  b.indices,
                                  b.indptr,
                                  c.indices,
                                  c.indptr,
                                  product_format_csr=True), c


def symbolic_csr_csc_gemm(a, b, product_format):
    """
    Assuming a is a instance of sparse.csr_matrix, b is a instance of sparse.csc_matrix.
    product_format can be either "csr" or "csc".

    Assuming the indices of a and b are all sorted.
    """
    assert product_format in {'csr', 'csc'}
    assert isinstance(a, sparse.csr_matrix)
    assert a.has_sorted_indices
    assert isinstance(b, sparse.csc_matrix)
    assert b.has_sorted_indices
    # The result will be in the following format:
    # [[(a_data_idx, b_data_idx), (a_, b_), (a_, b_)], [(a_, b_), (a_, b_)], [(a_, b_)]]
    # result[i] represent the multiply-accumulate needed for c.data[i]
    # result[i][j] represent the j-th multiply indices of a.data and b.data
    c = a @ sparse.csr_matrix(b)
    if product_format == 'csc':
        c = sparse.csc_matrix(c)
    c = c.sorted_indices()
    is_csr = (product_format == 'csr')
    return _symbolic_csr_csc_gemm(a.indices,
                                  a.indptr,
                                  b.indices,
                                  b.indptr,
                                  c.indices,
                                  c.indptr,
                                  product_format_csr=is_csr), c


def _test_case(product_format):
    m = random.randint(1, 100)
    n = random.randint(1, 100)
    k = random.randint(1, 100)
    a = sparse.random(m, k, format='csr')
    b = sparse.random(k, n, format='csc')
    result, c = symbolic_csr_csc_gemm(a, b, product_format=product_format)
    synthesized_c_data = []
    for i, pairs in enumerate(result):
        synthesized_c_data.append(0)
        for a_data_idx, b_data_idx in pairs:
            synthesized_c_data[-1] += a.data[a_data_idx] * b.data[b_data_idx]
    synthesized_c_data = np.array(synthesized_c_data)
    np.testing.assert_allclose(synthesized_c_data, c.data)


if __name__ == '__main__':
    for _ in range(100):
        _test_case('csr')
        _test_case('csc')
