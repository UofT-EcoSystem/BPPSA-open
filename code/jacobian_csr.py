import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import csr_matrix
from numba import jit, prange
import time
from jcb import jacobianT, jacobianT_sparse


@jit(nopython=True, nogil=True, parallel=True)
def compute_indptr_3d(n, d1, d2):
    num_row = d1 * n**2 + 1
    result = np.zeros(num_row)
    for i in prange(num_row):
        a = i // (n**2)
        b = i % (n**2)
        if b <= n:
            result[i] = a*d2*(9*n**2-6*n)+6*d2*b # yapf: disable
        elif b <= n * (n - 1):
            result[i] = a*d2*(9*n**2-6*n)+6*d2*n+9*d2*(b-n) # yapf: disable
        else:
            result[i] = a*d2*(9*n**2-6*n)+6*d2*n+9*d2*n*(n-2)+6*d2*(b-n*(n-1)) # yapf: disable
    return result


@jit(nopython=True, nogil=True, parallel=True)
def _compute_indptr_3d(c_i, c_o, h_i, w_i):
    indptr = np.zeros(c_i * h_i * w_i + 1)
    for i in prange(c_i * h_i * w_i + 1):
        a = np.floor(i / (h_i * w_i))
        b = i % (h_i * w_i)
        if b <= w_i:
            indptr[i] = a * c_o * (3 * w_i * (3 * h_i - 2)) + 6 * c_o * b
        elif b <= w_i * (h_i - 1):
            indptr[i] = a * c_o * (3 * w_i * (3 * h_i - 2)) + 6 * c_o * w_i + 9 * c_o * (b-w_i) # yapf: disable
        else:
            indptr[i] = a * c_o * (3 * w_i * (3 * h_i - 2)) + 6 * c_o * w_i + 9 * c_o * (w_i * (h_i - 2)) + 6 * c_o * (b - w_i * (h_i - 1)) # yapf: disable
    return indptr


@jit(nopython=True, nogil=True, parallel=True)
def compute_indices_3d(n, d1, d2, indptr):
    result = np.zeros((9 * n**2 - 6 * n) * d1 * d2)
    base = compute_base(n, d2)
    for i in prange(indptr.shape[0] - 1):
        start_i = int(indptr[i])
        end_i = int(indptr[i + 1])
        row_i = (base + i % (n * n) - (n + 1)) % (d2 * n * n)
        r = i % (n * n)
        if r < n:
            t1 = sub_array_1(row_i)
            t1.sort()
            result[start_i:end_i] = t1
        elif r >= n * (n - 1):
            t2 = sub_array_2(row_i)
            t2.sort()
            result[start_i:end_i] = t2
        else:
            t3 = row_i[:]
            t3.sort()
            result[start_i:end_i] = t3
    return result


@jit(nopython=True, nogil=True, parallel=True)
def _compute_indices_3d(c_i, c_o, h_i, w_i, indptr):
    base_const = np.array([-1, 0, 1])
    indices = np.zeros(3 * w_i * (3 * h_i - 2) * c_i * c_o)
    for i in prange(c_i * h_i * w_i):
        r = i % (h_i * w_i)
        base = np.zeros(9 * c_o)
        for j in prange(c_o):
            for k in prange(3):
                base[9*j+3*k : 9*j + 3*(k+1)] = (base_const + (j * h_i + k - 1) * w_i + r) % (c_o * h_i * w_i) # yapf: disable
        if r < w_i or r >= w_i * (h_i - 1):
            row = np.zeros(6 * c_o)
            left, right = (3, 9) if r < w_i else (0, 6)
            for j in prange(c_o):
                row[6 * j:6 * j + 6] = base[9 * j + left:9 * j + right]
        else:
            row = base
        row.sort()
        indices[indptr[i]:indptr[i + 1]] = row
    return indices


@jit(nopython=True, nogil=True, parallel=True)
def compute_data_3d(n, d1, d2, kernels, indptr):
    result = np.zeros((9 * n**2 - 6 * n) * d1 * d2)
    for i in prange(len(indptr) - 1):
        start_i = int(indptr[i])
        end_i = int(indptr[i + 1])
        m = i // (n * n)
        r = i % (n * n)
        data0 = kernels[:, m]
        start_w = 1
        if r < n * (n - 1):
            start_w = 0
        end_w = 3
        if r < n:
            end_w = 2
        data1 = data0[:, start_w:end_w]
        data2 = reverse_and_flatten(data1)
        result[start_i:end_i] = data2

        if i % n == 0:
            shifted_w = result[start_i:end_i]
            start_w = 0
            if i % (n * n) <= n:
                shifted_w = shift_left(shifted_w)
                start_w = 2
            shifted_w[start_w::3] = 0
            result[start_i:end_i] = shifted_w

        if i % n == n - 1:
            shifted_w = result[start_i:end_i]
            start_w = 2
            if i % (n * n) >= n * (n - 1) - 1:
                shifted_w = shift_right(shifted_w)
                start_w = 0
            shifted_w[start_w::3] = 0
            result[start_i:end_i] = shifted_w

    return result


@jit(nopython=True, nogil=True, parallel=True)
def _compute_data_3d(n, d1, d2, kernels, indptr):
    result = np.zeros((9 * n**2 - 6 * n) * d1 * d2)
    for i in prange(len(indptr) - 1):
        r = i % (n * n)
        m = int(np.floor(i / (n * n)))
        start_i = int(indptr[i])
        end_i = int(indptr[i + 1])
        if r < n:
            result[indptr[i]:indptr[i+1]] = kernels[:, m, 1::-1, ::-1].flatten() # yapf: disable
        elif r >= n * (n - 1):
            result[indptr[i]:indptr[i+1]] = kernels[:, m, 2:0:-1, ::-1].flatten() # yapf: disable
        else:
            result[indptr[i]:indptr[i+1]] = kernels[:, m, 2::-1, ::-1].flatten() # yapf: disable
        # Fix corner cases.
        if i % n == 0:
            shifted_w = result[start_i:end_i]
            start_w = 0
            if i % (n * n) <= n:
                shifted_w = shift_left(shifted_w)
                start_w = 2
            shifted_w[start_w::3] = 0
            result[start_i:end_i] = shifted_w

        if i % n == n - 1:
            shifted_w = result[start_i:end_i]
            start_w = 2
            if i % (n * n) >= n * (n - 1) - 1:
                shifted_w = shift_right(shifted_w)
                start_w = 0
            shifted_w[start_w::3] = 0
            result[start_i:end_i] = shifted_w

    return result


@jit(nopython=True, nogil=True, parallel=True)
def shift_left(w):
    length = w.shape[0]
    result = np.zeros(length)
    for i in prange(length - 1):
        result[i] = w[i + 1]
    result[-1] = w[0]
    return result


@jit(nopython=True, nogil=True, parallel=True)
def shift_right(w):
    length = w.shape[0]
    result = np.zeros(length)
    for i in prange(1, length):
        result[i] = w[i - 1]
    result[0] = w[-1]
    return result


@jit(nopython=True, nogil=True, parallel=True)
def compute_base(n, d2):
    result = np.zeros(9 * d2)
    for i in prange(d2):
        for j in prange(3):
            result[9 * i + 3 * j:9 * i + 3 * (j + 1)] = np.array(
                [i * n**2 + j * n, i * n**2 + j * n + 1, i * n**2 + j * n + 2])
    return result


@jit(nopython=True, nogil=True, parallel=True)
def sub_array_1(arr):
    length = arr.shape[0] * 2 // 3
    result = np.zeros(length)
    for i in prange(len(arr)):
        m = i // 9
        r = i % 9
        if r >= 3:
            result[6 * m + r - 3] = arr[i]
    return result


@jit(nopython=True, nogil=True, parallel=True)
def sub_array_2(arr):
    length = arr.shape[0] * 2 // 3
    result = np.zeros(length)
    for i in prange(len(arr)):
        m = i // 9
        r = i % 9
        if r <= 5:
            result[6 * m + r] = arr[i]
    return result


@jit(nopython=True, nogil=True, parallel=True)
def reverse_and_flatten(arr):
    length = arr.shape[0]
    kernel_size = arr.shape[1] * arr.shape[2]
    result = np.zeros(length * kernel_size)
    for i in prange(length):
        for j in prange(arr.shape[1]):
            for k in prange(arr.shape[2]):
                result[i * kernel_size + j * arr.shape[2] + k] = arr[i, arr.shape[1] - 1 - j, arr.shape[2] - 1 - k] # yapf: disable
    return result


def jacobianT_conv2d(module, x):
    """ Expects module to be the nn.Conv2d instance.
    Expects x to be a tensor of size (B, C, H, W).
    """
    with torch.no_grad():
        weight = module.weight.detach().numpy()
        hw = x.size(3)
    d2, d1 = weight.shape[0], weight.shape[1]
    #indptr = compute_indptr_3d(hw, d1, d2).astype(np.int32)
    indptr = _compute_indptr_3d(d1, d2, hw, hw).astype(np.int32)
    #indices = compute_indices_3d(hw, d1, d2, indptr).astype(np.int32)
    indices = _compute_indices_3d(d1, d2, hw, hw, indptr).astype(np.int32)
    #data = compute_data_3d(hw, d1, d2, weight, indptr).astype(np.float32)
    data = _compute_data_3d(hw, d1, d2, weight, indptr).astype(np.float32)
    ret = csr_matrix((data, indices, indptr))
    ret.eliminate_zeros()
    return ret


@jit(nopython=True, nogil=True, parallel=True)
def _jacobianT_relu_indptr(length):
    return np.arange(length + 1)


@jit(nopython=True, nogil=True, parallel=True)
def _jacobianT_relu_indices(length):
    return np.arange(length)


@jit(nopython=True, nogil=True, parallel=True)
def _jacobianT_relu_data(x_flatten):
    length = x_flatten.shape[0]
    data = np.zeros(length)
    for i in prange(length):
        if x_flatten[i] > 0.0:
            data[i] = 1.0
        else:
            data[i] = 0.0
    return data


def jacobianT_relu(x):
    """ Expects x to be a tensor of size (B, C, H, W).
    """
    # For a single sample.
    x = x.detach().numpy()
    batch_size = x.shape[0]
    ret = []
    for sample_idx in range(batch_size):
        x_flatten = x[sample_idx].flatten()
        x_length = x_flatten.shape[0]
        indptr = _jacobianT_relu_indptr(x_length).astype(np.int32)
        indices = _jacobianT_relu_indices(x_length).astype(np.int32)
        data = _jacobianT_relu_data(x_flatten).astype(np.float32)
        jcbT = csr_matrix((data, indices, indptr))
        jcbT.eliminate_zeros()
        ret.append(jcbT)
    return ret


@jit(nopython=True, nogil=True, parallel=True)
def _jacobianT_maxpool2d_mapping(pool_indices, channels, x_h, x_w, y_h, y_w):
    mapping = np.full(channels * x_h * x_w, -1)
    for c in prange(pool_indices.shape[0]):
        for h in prange(pool_indices.shape[1]):
            for w in prange(pool_indices.shape[2]):
                i = c * x_h * x_w + pool_indices[c, h, w]
                j = (c * y_h + h) * y_w + w
                mapping[i] = j
    return mapping


@jit(nopython=True, nogil=True, parallel=True)
def _jacobianT_maxpool2d_indptr(mapping):
    x_length = mapping.shape[0]
    indptr = np.arange(x_length + 1)
    ptr = 0
    for i in range(x_length):
        indptr[i] = ptr
        if mapping[i] != -1:
            ptr += 1
    indptr[-1] = ptr
    return indptr


@jit(nopython=True, nogil=True, parallel=True)
def _jacobianT_maxpool2d_indices(mapping, channels, y_h, y_w):
    indices = np.zeros(channels * y_h * y_w)
    indices_ptr = 0
    for i in range(mapping.shape[0]):
        if mapping[i] == -1:
            continue
        indices[indices_ptr] = mapping[i]
        indices_ptr += 1
    return indices


@jit(nopython=True, nogil=True, parallel=True)
def _jacobianT_maxpool2d_data(y_length):
    return np.ones(y_length)


def jacobianT_maxpool2d(module, x):
    """ Expects module to be a nn.MaxPool2d instance.
    Expects x to be a torch tensor of shape (B, C, H, W).

    Expects no overlap in pooling.
    """
    with torch.no_grad():
        y, pool_indices = F.max_pool2d(
            x,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            ceil_mode=module.ceil_mode,
            return_indices=True,
        )
        x, y, pool_indices = x.detach().numpy(), y.numpy(), pool_indices.numpy(
        )
    x_h, x_w = x.shape[2], x.shape[3]
    batch_size, channels, y_h, y_w = y.shape
    y_length = channels * y_h * y_w
    ret = []
    for sample_idx in range(batch_size):
        mapping = _jacobianT_maxpool2d_mapping(pool_indices[sample_idx],
                                               channels, x_h, x_w, y_h, y_w)
        indptr = _jacobianT_maxpool2d_indptr(mapping).astype(np.int32)
        indices = _jacobianT_maxpool2d_indices(mapping, channels, y_h,
                                               y_w).astype(np.int32)
        data = _jacobianT_maxpool2d_data(y_length).astype(np.float32)
        jcbT = csr_matrix((data, indices, indptr))
        jcbT.eliminate_zeros()
        ret.append(jcbT)
    return ret


def jacobianT_linear(module):
    with torch.no_grad():
        weight = module.weight.detach().numpy()
        ret = csr_matrix(weight.T)
        ret.eliminate_zeros()
        return ret


def main(args):
    cfg = [64, 'M']  # The first three ops of VGG-11.
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            torch.nn.init.ones_(conv2d.weight)
            layers += [conv2d, nn.ReLU(inplace=False)]
            in_channels = v
    x = torch.rand(args.x_shape, requires_grad=True)

    def test_jcbT_conv2d(expected, actual):
        actual.check_format()
        expected = csr_matrix(expected[0])
        np.testing.assert_equal(expected.indptr, actual.indptr)
        np.testing.assert_equal(expected.indices, actual.indices)
        np.testing.assert_equal(expected.data, actual.data)
        assert expected.nnz == actual.nnz
        print('{}:{}'.format(expected.indptr.dtype, actual.indptr.dtype))
        print('{}:{}'.format(expected.indices.dtype, actual.indices.dtype))
        print('{}:{}'.format(expected.data.dtype, actual.data.dtype))

    def test_jcbT_relu_maxpool2d(expected, actual):
        for sample_idx in range(expected.shape[0]):
            e = csr_matrix(expected[sample_idx])
            a = actual[sample_idx]
            a.check_format()
            np.testing.assert_equal(e.indptr, a.indptr)
            np.testing.assert_equal(e.indices, a.indices)
            np.testing.assert_equal(e.data, a.data)
            assert e.nnz == a.nnz
            print('{}:{}'.format(e.indptr.dtype, a.indptr.dtype))
            print('{}:{}'.format(e.indices.dtype, a.indices.dtype))
            print('{}:{}'.format(e.data.dtype, a.data.dtype))

    for layer in layers:
        y = layer(x)
        layer_is_conv2d = isinstance(layer, nn.Conv2d)
        layer_is_relu = isinstance(layer, nn.ReLU)
        layer_is_maxpool2d = isinstance(layer, nn.MaxPool2d)
        layer_is_linear = isinstance(layer, nn.Linear)
        if layer_is_conv2d or layer_is_relu or layer_is_maxpool2d:
            if layer_is_conv2d:
                print_header = lambda: print('======= Conv2d: =======')
                run_jcbT = lambda: jacobianT_conv2d(layer, x)
                test_jcbT = test_jcbT_conv2d
            elif layer_is_relu:
                print_header = lambda: print('======= ReLU: =======')
                run_jcbT = lambda: jacobianT_relu(x)
                test_jcbT = test_jcbT_relu_maxpool2d
            elif layer_is_maxpool2d:
                print_header = lambda: print('======= MaxPool2d: =======')
                run_jcbT = lambda: jacobianT_maxpool2d(layer, x)
                test_jcbT = test_jcbT_relu_maxpool2d
            else:
                raise RuntimeError('Impossible to reach here!')
            print_header()
            tic = time.perf_counter()
            actual = run_jcbT()
            latency = time.perf_counter() - tic

            if args.compare:
                # The previous run is for compile statically.
                num_trial = 1000
                tic = time.perf_counter()
                for _ in range(num_trial):
                    run_jcbT()
                latency = (time.perf_counter() - tic) / num_trial

                tic = time.perf_counter()
                expected = jacobianT(x, y).numpy()
                latency_baseline = time.perf_counter() - tic
                test_jcbT(expected, actual)
                print('Generation Speedup: {} / {} = {}'.format(
                    latency_baseline, latency, latency_baseline / latency))
            else:
                print(latency)
        x = y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--compare',
        dest='compare',
        action='store_true',
        help='Compare to get Jacobian analytically v.s. via Autograd trick.',
    )
    parser.add_argument(
        '-nc',
        '--no-compare',
        dest='compare',
        action='store_false',
        help='Do not compare.',
    )
    parser.set_defaults(compare=True)
    parser.add_argument(
        '-s',
        '--x_shape',
        nargs=4,
        type=int,
        required=False,
        default=[1, 3, 32, 32],
        help='The shape of the input tensor.',
    )
    main(parser.parse_args())
