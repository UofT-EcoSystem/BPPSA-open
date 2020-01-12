#include <math.h>
#include <torch/extension.h>

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TENSOR_ACCESSOR(dim) \
  torch::PackedTensorAccessor<scalar_t, dim, torch::RestrictPtrTraits, size_t>
#define GET_TENSOR_ACCESSOR(tensor, dim) \
  tensor.packed_accessor<scalar_t, dim, torch::RestrictPtrTraits, size_t>()

// Calculate pow(2, p)
__host__ __device__ __forceinline__ size_t pow2(size_t p) { return 1 << p; }
// Calculate x % pow(2, p)
__host__ __device__ __forceinline__ size_t mod_pow2(size_t x, size_t p) {
  // Shift x to the left for 8 * sizeof(size_t) - p bits. Shift the rest back
  // and returns.
  const size_t shift = 8 * sizeof(size_t) - p;
  return (x << shift) >> shift;
}
// Calculate ceil(x / pow(2, p))
__host__ __device__ __forceinline__ size_t ceil_div_pow2(size_t x, size_t p) {
  // Shift x to the right for (p) bits.
  const size_t res = x >> p;
  return mod_pow2(x, p) ? res + 1 : res;
}
// Calculate floor(x / pow(2, p))
__host__ __device__ __forceinline__ size_t floor_div_pow2(size_t x, size_t p) {
  return x >> p;
}

template <typename scalar_t>
__device__ __forceinline__ void mvmul(size_t l, size_t r, size_t write_back_idx,
                                      size_t sample_idx, size_t jcbT_dim,
                                      TENSOR_ACCESSOR(4) inputs,
                                      scalar_t* shared_mem) {
  size_t w = threadIdx.x;
  scalar_t* A_T = shared_mem;
  scalar_t* B = shared_mem + jcbT_dim * (jcbT_dim + 1);
  for (size_t k = 0; k < jcbT_dim; ++k) {
    A_T[w * (jcbT_dim + 1) + k] = inputs[l][sample_idx][k][w];
  }
  B[w] = inputs[r][sample_idx][0][w];
  scalar_t C = (scalar_t)0;
  __syncthreads();
  // Calculate C[target].
  for (size_t k = 0; k < jcbT_dim; ++k) {
    C += A_T[k * (jcbT_dim + 1) + w] * B[k];
  }
  // Write C back.
  inputs[write_back_idx][sample_idx][0][w] = C;
}

template <typename scalar_t>
__device__ __forceinline__ void mmmul(size_t l, size_t r, bool is_r_vec,
                                      size_t write_back_idx, size_t sample_idx,
                                      size_t jcbT_dim,
                                      TENSOR_ACCESSOR(4) inputs,
                                      scalar_t* shared_mem) {
  scalar_t* A = shared_mem;
  scalar_t* B = shared_mem + jcbT_dim * jcbT_dim;
  // scalar_t* C = shared_mem + 2 * jcbT_dim * jcbT_dim;
  // Coorperatively load inputs[l, sample_idx, :, :] into A; load
  // inputs[r, sample_idx, :, :] into B. Clear C.
  size_t h = threadIdx.y, w = threadIdx.x;
  size_t target = h * jcbT_dim + w;
  A[target] = inputs[l][sample_idx][h][w];
  if (is_r_vec) {
    if (h == 0) {
      B[target] = inputs[r][sample_idx][h][w];
    }
  } else {
    B[target] = inputs[r][sample_idx][h][w];
  }
  // C[target] = (scalar_t)0;
  scalar_t C = (scalar_t)0;
  __syncthreads();
  // Calculate C[target].
  if (is_r_vec) {
    if (h == 0) {
      for (size_t k = 0; k < jcbT_dim; ++k) {
        C += A[w * jcbT_dim + k] * B[k];
      }
    }
  } else {
    for (size_t k = 0; k < jcbT_dim; ++k) {
      C += A[h * jcbT_dim + k] * B[k * jcbT_dim + w];
    }
  }
  //__syncthreads();
  // Write C back.
  inputs[write_back_idx][sample_idx][h][w] = C;
}

template <typename scalar_t>
__device__ __forceinline__ void down_sweep_op(size_t l, size_t r, bool is_r_I,
                                              size_t sample_idx,
                                              size_t jcbT_dim,
                                              TENSOR_ACCESSOR(4) inputs,
                                              scalar_t* shared_mem) {
  if (!is_r_I) {
    mvmul(l, r, r, sample_idx, jcbT_dim, inputs, shared_mem);
    // At current point, the orig. r is in B, therefore we write B to l.
    scalar_t* B = shared_mem + jcbT_dim * (jcbT_dim + 1);
    inputs[l][sample_idx][0][threadIdx.x] = B[threadIdx.x];
  } else {
    // A * I = A; therefore, copying the elements of A from l to r instead.
    inputs[r][sample_idx][0][threadIdx.x] =
        inputs[l][sample_idx][0][threadIdx.x];
  }
}

template <typename scalar_t>
__global__ void up_sweep_kernel(size_t d, size_t n, size_t jcbT_dim,
                                TENSOR_ACCESSOR(4) inputs) {
  extern __shared__ int shared_mem_untyped[];
  scalar_t* shared_mem = (scalar_t*)shared_mem_untyped;
  size_t i = blockIdx.x;
  i = i * pow2(d + 1);
  size_t l = i + pow2(d) - 1;
  size_t r = i + pow2(d + 1) - 1;
  r = r < n ? r : n;
  mmmul(r, l, i == 0, r, blockIdx.y, jcbT_dim, inputs, shared_mem);
}

template <typename scalar_t>
__global__ void down_sweep_kernel(size_t d, size_t n, size_t jcbT_dim,
                                  TENSOR_ACCESSOR(4) inputs) {
  extern __shared__ int shared_mem_untyped[];
  scalar_t* shared_mem = (scalar_t*)shared_mem_untyped;
  size_t i = blockIdx.x;
  i = i * pow2(d + 1);
  size_t l = i + pow2(d) - 1;
  size_t r = i + pow2(d + 1) - 1;
  r = r < n ? r : n;
  down_sweep_op(l, r, i == 0, blockIdx.y, jcbT_dim, inputs, shared_mem);
}

template <typename scalar_t>
__global__ void move_results_kernel(const TENSOR_ACCESSOR(4) inputs,
                                    TENSOR_ACCESSOR(3) results) {
  // Copy the resulted vectors from inputs to results.
  if (blockIdx.x > 0) {  // results[0] should be I.
    results[blockIdx.x][blockIdx.y][threadIdx.x] =
        inputs[blockIdx.x][blockIdx.y][0][threadIdx.x];
  }
}

inline size_t ceil_log(size_t x) {
  return static_cast<size_t>(ceil(log2(static_cast<double>(x))));
}

#define CHECK_CUDA_ERROR()              \
  cudaDeviceSynchronize();              \
  cudaError_t err = cudaGetLastError(); \
  if (err != cudaSuccess) {             \
    std::stringstream ss;               \
    ss << "CUDA runtime error " << err; \
    throw std::runtime_error(ss.str()); \
  }

// inputs[0, :, :, 0] assumes to be the gradient vectors.
// inputs[0, :, :, 1:] are all zeros.
torch::Tensor scan_cuda(torch::Tensor inputs) {
  size_t n = inputs.size(0) - 1;
  size_t num_levels = ceil_log(n + 1) - 2 + 1;
  size_t batch_size = inputs.size(1);
  size_t jcbT_dim = inputs.size(2);

  auto results = torch::zeros({inputs.size(0), inputs.size(1), inputs.size(2)},
                              inputs.options());

  const dim3 threads(jcbT_dim, jcbT_dim);
  for (size_t d = 0; d < num_levels; ++d) {
    const dim3 blocks(ceil_div_pow2(n - pow2(d) + 1, d + 1), batch_size);
    AT_DISPATCH_FLOATING_TYPES(inputs.type(), "up_sweep_kernel", [&] {
      up_sweep_kernel<scalar_t>
          <<<blocks, threads, 2 * sizeof(scalar_t) * jcbT_dim * jcbT_dim>>>(
              d, n, jcbT_dim, GET_TENSOR_ACCESSOR(inputs, 4));
    });
  }
  // a[n] = I
  for (int d = num_levels; d >= 0; --d) {
    const dim3 blocks(ceil_div_pow2(n - pow2(d) + 1, d + 1), batch_size);
    AT_DISPATCH_FLOATING_TYPES(inputs.type(), "down_sweep_kernel", [&] {
      down_sweep_kernel<scalar_t>
          <<<blocks, jcbT_dim,
             1 * sizeof(scalar_t) * jcbT_dim*(jcbT_dim + 2)>>>(
              d, n, jcbT_dim, GET_TENSOR_ACCESSOR(inputs, 4));
    });
  }
  const dim3 blocks(n + 1, batch_size);
  AT_DISPATCH_FLOATING_TYPES(inputs.type(), "move_results_kernel", [&] {
    move_results_kernel<scalar_t><<<blocks, threads>>>(
        GET_TENSOR_ACCESSOR(inputs, 4), GET_TENSOR_ACCESSOR(results, 3));
  });
  return results;
}

template <typename scalar_t>
__global__ void copy_weightT_to_inputs(size_t idx_in_inputs,
                                       TENSOR_ACCESSOR(4) inputs,
                                       const TENSOR_ACCESSOR(2) weight) {
  inputs[idx_in_inputs][blockIdx.x][threadIdx.x][threadIdx.y] =
      weight[threadIdx.y][threadIdx.x];
}

template <typename scalar_t>
__global__ void set_tanh_jcbT_in_inputs(size_t idx_in_inputs,
                                        TENSOR_ACCESSOR(4) inputs,
                                        const TENSOR_ACCESSOR(2) hx) {
  // inputs[idx_in_inputs][blockIdx.x][0][threadIdx.x] = 0.0;
  scalar_t tanh_z = hx[blockIdx.x][threadIdx.x];
  inputs[idx_in_inputs][blockIdx.x][threadIdx.x][threadIdx.x] =
      1.0 - tanh_z * tanh_z;
}

void fill_inputs_cuda(torch::Tensor inputs, torch::Tensor weight_hh,
                      torch::Tensor hx, size_t i) {
  size_t batch_size = inputs.size(1);
  size_t jcbT_dim = inputs.size(2);
  size_t idx = inputs.size(0) - 1 - 2 * i;
  {
    dim3 blocks(batch_size);
    dim3 threads(jcbT_dim, jcbT_dim);
    AT_DISPATCH_FLOATING_TYPES(inputs.type(), "copy_weightT_to_inputs", [&] {
      copy_weightT_to_inputs<scalar_t>
          <<<blocks, threads>>>(idx, GET_TENSOR_ACCESSOR(inputs, 4),
                                GET_TENSOR_ACCESSOR(weight_hh, 2));
    });
  }
  {
    idx -= 1;
    dim3 blocks(batch_size);
    dim3 threads(jcbT_dim);
    AT_DISPATCH_FLOATING_TYPES(inputs.type(), "set_tanh_jcbT_in_inputs", [&] {
      set_tanh_jcbT_in_inputs<scalar_t><<<blocks, threads>>>(
          idx, GET_TENSOR_ACCESSOR(inputs, 4), GET_TENSOR_ACCESSOR(hx, 2));
    });
  }
}

void fill_last_dense_layer_weight_to_inputs_cuda(torch::Tensor inputs,
                                                 torch::Tensor weight) {
  size_t batch_size = inputs.size(1);
  size_t jcbT_dim = inputs.size(2);
  dim3 blocks(batch_size);
  dim3 threads(jcbT_dim, jcbT_dim);
  AT_DISPATCH_FLOATING_TYPES(inputs.type(), "copy_weightT_to_inputs", [&] {
    copy_weightT_to_inputs<scalar_t><<<blocks, threads>>>(
        1, GET_TENSOR_ACCESSOR(inputs, 4), GET_TENSOR_ACCESSOR(weight, 2));
  });
}

template <typename scalar_t>
__global__ void reverse_seq_cuda_kernel(size_t seq_len,
                                        TENSOR_ACCESSOR(3) target,
                                        const TENSOR_ACCESSOR(3) source) {
  target[blockIdx.x][threadIdx.x][threadIdx.y] =
      source[seq_len - 1 - blockIdx.x][threadIdx.x][threadIdx.y];
}

template <typename scalar_t>
__global__ void reverse_seq_cuda_kernel_large_batch_size(
    size_t seq_len, size_t batch_size, size_t group_factor,
    TENSOR_ACCESSOR(3) target, const TENSOR_ACCESSOR(3) source) {
  size_t sample_idx = blockIdx.y * group_factor + threadIdx.x;
  if (sample_idx >= batch_size) {
    return;
  }
  target[blockIdx.x][sample_idx][threadIdx.y] =
      source[seq_len - 1 - blockIdx.x][sample_idx][threadIdx.y];
}

void reverse_seq_cuda(torch::Tensor target, torch::Tensor source) {
  size_t seq_len = source.size(0);
  size_t batch_size = source.size(1);
  size_t hidden_size = source.size(2);
  size_t group_factor = 32;
  if (batch_size <= group_factor) {
    dim3 blocks(seq_len);
    dim3 threads(batch_size, hidden_size);
    AT_DISPATCH_FLOATING_TYPES(source.type(), "reverse_seq_cuda_kernel", [&] {
      reverse_seq_cuda_kernel<scalar_t>
          <<<blocks, threads>>>(seq_len, GET_TENSOR_ACCESSOR(target, 3),
                                GET_TENSOR_ACCESSOR(source, 3));
    });
  } else {
    dim3 blocks(seq_len,
                static_cast<size_t>(ceil(static_cast<double>(batch_size) /
                                         static_cast<double>(group_factor))));
    dim3 threads(group_factor, hidden_size);
    AT_DISPATCH_FLOATING_TYPES(
        source.type(), "reverse_seq_cuda_kernel_large_batch_size", [&] {
          reverse_seq_cuda_kernel_large_batch_size<scalar_t>
              <<<blocks, threads>>>(seq_len, batch_size, group_factor,
                                    GET_TENSOR_ACCESSOR(target, 3),
                                    GET_TENSOR_ACCESSOR(source, 3));
        });
  }
}

template <typename scalar_t>
__global__ void copy_weightT_to_inputs2(size_t n, TENSOR_ACCESSOR(4) inputs,
                                        const TENSOR_ACCESSOR(2) weight_hh) {
  size_t i = n - 1 - 2 * blockIdx.x;
  inputs[i][blockIdx.y][threadIdx.x][threadIdx.y] =
      weight_hh[threadIdx.y][threadIdx.x];
}

template <typename scalar_t>
__global__ void set_tanh_jcbT_in_inputs2(size_t n, TENSOR_ACCESSOR(4) inputs,
                                         const TENSOR_ACCESSOR(3) hx_s) {
  size_t i = n - 1 - 2 * blockIdx.x - 1;
  // inputs[i][threadIdx.x][0][threadIdx.y] = 0.0;
  scalar_t tanh_z = hx_s[blockIdx.x][threadIdx.x][threadIdx.y];
  inputs[i][threadIdx.x][threadIdx.y][threadIdx.y] = 1.0 - tanh_z * tanh_z;
}

template <typename scalar_t>
__global__ void set_tanh_jcbT_in_inputs2_large_batch_size(
    size_t n, size_t batch_size, size_t group_factor, TENSOR_ACCESSOR(4) inputs,
    const TENSOR_ACCESSOR(3) hx_s) {
  size_t sample_idx = blockIdx.y * group_factor + threadIdx.x;
  if (sample_idx >= batch_size) {
    return;
  }
  size_t i = n - 1 - 2 * blockIdx.x - 1;
  // inputs[i][sample_idx][0][threadIdx.y] = 0.0;
  scalar_t tanh_z = hx_s[blockIdx.x][sample_idx][threadIdx.y];
  inputs[i][sample_idx][threadIdx.y][threadIdx.y] = 1.0 - tanh_z * tanh_z;
}

void fill_inputs2_cuda(torch::Tensor inputs, torch::Tensor weight_hh,
                       torch::Tensor hx_s) {
  size_t n = inputs.size(0);
  size_t seq_len = hx_s.size(0);
  size_t batch_size = inputs.size(1);
  size_t jcbT_dim = inputs.size(2);
  size_t group_factor = 32;

  {
    dim3 blocks(seq_len, batch_size);
    dim3 threads(jcbT_dim, jcbT_dim);
    AT_DISPATCH_FLOATING_TYPES(inputs.type(), "copy_weightT_to_inputs2", [&] {
      copy_weightT_to_inputs2<scalar_t><<<blocks, threads>>>(
          n, GET_TENSOR_ACCESSOR(inputs, 4), GET_TENSOR_ACCESSOR(weight_hh, 2));
    });
  }
  {
    if (batch_size <= group_factor) {
      dim3 blocks(seq_len);
      dim3 threads(batch_size, jcbT_dim);
      AT_DISPATCH_FLOATING_TYPES(
          inputs.type(), "set_tanh_jcbT_in_inputs2", [&] {
            set_tanh_jcbT_in_inputs2<scalar_t>
                <<<blocks, threads>>>(n, GET_TENSOR_ACCESSOR(inputs, 4),
                                      GET_TENSOR_ACCESSOR(hx_s, 3));
          });
    } else {
      dim3 blocks(seq_len,
                  static_cast<size_t>(ceil(static_cast<double>(batch_size) /
                                           static_cast<double>(group_factor))));
      dim3 threads(group_factor, jcbT_dim);
      AT_DISPATCH_FLOATING_TYPES(
          inputs.type(), "set_tanh_jcbT_in_inputs2_large_batch_size", [&] {
            set_tanh_jcbT_in_inputs2_large_batch_size<scalar_t>
                <<<blocks, threads>>>(n, batch_size, group_factor,
                                      GET_TENSOR_ACCESSOR(inputs, 4),
                                      GET_TENSOR_ACCESSOR(hx_s, 3));
          });
    }
  }
}
