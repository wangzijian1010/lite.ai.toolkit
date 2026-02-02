//
// YOLO 预处理 CUDA Kernel
// 功能: Resize + Letterbox + BGR2RGB + Normalize + HWC2CHW
// 一次 kernel 完成所有预处理，避免多次内存拷贝
//

#ifndef LITE_AI_TOOLKIT_PREPROCESS_CUH
#define LITE_AI_TOOLKIT_PREPROCESS_CUH

#include <cuda_runtime.h>
#include <cstdint>

// 预处理参数结构体
struct PreprocessParams {
    int src_width;      // 原图宽度
    int src_height;     // 原图高度
    int dst_width;      // 目标宽度 (如 640)
    int dst_height;     // 目标高度 (如 640)
    float scale;        // 缩放比例
    int pad_left;       // 左边 padding
    int pad_top;        // 上边 padding
    int new_width;      // 缩放后宽度 (不含 padding)
    int new_height;     // 缩放后高度 (不含 padding)
    float mean[3];      // 归一化均值 (通常 0)
    float std[3];       // 归一化标准差 (通常 255)
};

// 计算 letterbox 参数
__host__ void compute_letterbox_params(
    int src_width, int src_height,
    int dst_width, int dst_height,
    PreprocessParams& params
);

// 主预处理 kernel
// 输入: uint8 BGR HWC [H, W, 3]
// 输出: float32 RGB CHW [3, dst_H, dst_W]
__global__ void preprocess_kernel(
    const uint8_t* __restrict__ src,    // 输入图像 (GPU 内存)
    float* __restrict__ dst,             // 输出张量 (GPU 内存)
    PreprocessParams params
);

// 双线性插值版本 (质量更好)
__global__ void preprocess_bilinear_kernel(
    const uint8_t* __restrict__ src,
    float* __restrict__ dst,
    PreprocessParams params
);

#endif //LITE_AI_TOOLKIT_PREPROCESS_CUH
