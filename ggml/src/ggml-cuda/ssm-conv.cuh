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

struct ggml_cuda_dilated_conv_history_args {
    ggml_cuda_dilated_conv_args conv;
    const ggml_tensor * history;
    const ggml_tensor * current;
    ggml_tensor * state;
    int slots;
    int64_t slot_stride;
};

void ggml_cuda_op_dilated_conv_history(ggml_backend_cuda_context & ctx, const ggml_cuda_dilated_conv_history_args & args);
