#include <torch/extension.h>

// inputs must be a [chain_len, batch_size, J_h, J_w] tensor and J_h == J_w;
// inputs[0, :, :, :] must be the gradient vector.
torch::Tensor scan_cuda(torch::Tensor inputs);
void fill_inputs_cuda(torch::Tensor inputs, torch::Tensor weight_hh,
                      torch::Tensor hx, size_t i);
void fill_last_dense_layer_weight_to_inputs_cuda(torch::Tensor inputs,
                                                 torch::Tensor weight);
void reverse_seq_cuda(torch::Tensor target, torch::Tensor source);
void fill_inputs2_cuda(torch::Tensor inputs, torch::Tensor weight_hh,
                       torch::Tensor hx_s);

#define CHECK_CUDA(x) \
  AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA_CONTIGUOUS(x) \
  CHECK_CUDA(x);                 \
  CHECK_CONTIGUOUS(x)
#define CHECK_DIM(x, d) AT_ASSERTM(x.dim() == d, #x " must be dim " #d)

torch::Tensor scan(torch::Tensor inputs) {
  CHECK_CUDA_CONTIGUOUS(inputs);
  CHECK_DIM(inputs, 4);
  AT_ASSERTM(inputs.size(2) == inputs.size(3),
             " inputs.size(2) != inputs.size(3)");
  return scan_cuda(inputs);
}

void fill_inputs(torch::Tensor inputs, torch::Tensor weight_hh,
                 torch::Tensor hx, size_t i) {
  CHECK_CUDA_CONTIGUOUS(inputs);
  CHECK_DIM(inputs, 4);
  AT_ASSERTM(inputs.size(2) == inputs.size(3),
             " inputs.size(2) != inputs.size(3)");
  fill_inputs_cuda(inputs, weight_hh, hx, i);
}

void fill_last_dense_layer_weight_to_inputs(torch::Tensor inputs,
                                            torch::Tensor weight) {
  CHECK_CUDA_CONTIGUOUS(inputs);
  CHECK_DIM(inputs, 4);
  AT_ASSERTM(inputs.size(2) == inputs.size(3),
             " inputs.size(2) != inputs.size(3)");
  fill_last_dense_layer_weight_to_inputs_cuda(inputs, weight);
}

void reverse_seq(torch::Tensor target, torch::Tensor source) {
  CHECK_CUDA_CONTIGUOUS(source);
  CHECK_CUDA_CONTIGUOUS(target);
  CHECK_DIM(source, 3);
  CHECK_DIM(target, 3);
  AT_ASSERTM(source.size(0) == target.size(0),
             " source.size(0) != target.size(0)");
  AT_ASSERTM(source.size(1) == target.size(1),
             " source.size(1) != target.size(1)");
  AT_ASSERTM(source.size(2) == target.size(2),
             " source.size(2) != target.size(2)");
  reverse_seq_cuda(target, source);
}

void fill_inputs2(torch::Tensor inputs, torch::Tensor weight_hh,
                  torch::Tensor hx_s) {
  CHECK_CUDA_CONTIGUOUS(inputs);
  CHECK_CUDA_CONTIGUOUS(weight_hh);
  CHECK_CUDA_CONTIGUOUS(hx_s);
  CHECK_DIM(inputs, 4);
  CHECK_DIM(weight_hh, 2);
  CHECK_DIM(hx_s, 3);
  AT_ASSERTM(inputs.size(2) == inputs.size(3),
             " inputs.size(2) != inputs.size(3)");
  fill_inputs2_cuda(inputs, weight_hh, hx_s);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scan", &scan, "scan (CUDA)");
  m.def("fill_inputs", &fill_inputs, "fill_inputs (CUDA)");
  m.def("fill_last_dense_layer_weight_to_inputs",
        &fill_last_dense_layer_weight_to_inputs,
        "fill_last_dense_layer_weight_to_inputs (CUDA)");
  m.def("reverse_seq", &reverse_seq, "reverse_seq (CUDA)");
  m.def("fill_inputs2", &fill_inputs2, "fill_inputs2 (CUDA)");
}
