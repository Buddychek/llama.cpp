#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

struct ggml_cuda_dilated_conv_args {
    const ggml_tensor * source;
    const ggml_tensor * weights;
    ggml_tensor * dst;
    const ggml_tensor * residual[2];
    int64_t first_row;
    int64_t dilation;
    bool apply_silu;
    int other_first;
};

void ggml_cuda_op_dilated_conv(ggml_backend_cuda_context & ctx, const ggml_cuda_dilated_conv_args & args);
