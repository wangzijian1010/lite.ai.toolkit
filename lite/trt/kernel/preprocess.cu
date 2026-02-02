//
// YOLO 预处理 CUDA Kernel 实现
//

#include "preprocess.cuh"

// 计算 letterbox 参数 (CPU 端)
__host__ void compute_letterbox_params(
    int src_width, int src_height,
    int dst_width, int dst_height,
    PreprocessParams& params
) {
    params.src_width = src_width;
    params.src_height = src_height;
    params.dst_width = dst_width;
    params.dst_height = dst_height;
    
    // 计算缩放比例 (保持宽高比)
    float scale_w = (float)dst_width / src_width;
    float scale_h = (float)dst_height / src_height;
    params.scale = fminf(scale_w, scale_h);
    
    // 缩放后的尺寸
    params.new_width = (int)(src_width * params.scale);
    params.new_height = (int)(src_height * params.scale);
    
    // padding (居中)
    params.pad_left = (dst_width - params.new_width) / 2;
    params.pad_top = (dst_height - params.new_height) / 2;
    
    // 默认归一化参数: /255
    params.mean[0] = params.mean[1] = params.mean[2] = 0.0f;
    params.std[0] = params.std[1] = params.std[2] = 255.0f;
}

// 最近邻插值预处理 kernel (速度快)
__global__ void preprocess_kernel(
    const uint8_t* __restrict__ src,
    float* __restrict__ dst,
    PreprocessParams params
) {
    // 目标图像坐标
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x >= params.dst_width || dst_y >= params.dst_height) return;
    
    // 输出位置 (CHW 格式)
    int dst_area = params.dst_width * params.dst_height;
    
    float r, g, b;
    
    // 检查是否在有效区域内 (非 padding 区域)
    int x_in_resized = dst_x - params.pad_left;
    int y_in_resized = dst_y - params.pad_top;
    
    if (x_in_resized >= 0 && x_in_resized < params.new_width &&
        y_in_resized >= 0 && y_in_resized < params.new_height) {
        
        // 映射回原图坐标 (最近邻)
        int src_x = (int)(x_in_resized / params.scale);
        int src_y = (int)(y_in_resized / params.scale);
        
        // 边界检查
        src_x = min(max(src_x, 0), params.src_width - 1);
        src_y = min(max(src_y, 0), params.src_height - 1);
        
        // 读取 BGR 像素
        int src_idx = (src_y * params.src_width + src_x) * 3;
        b = src[src_idx + 0];
        g = src[src_idx + 1];
        r = src[src_idx + 2];
    } else {
        // padding 区域填充 114 (YOLO 标准)
        r = g = b = 114.0f;
    }
    
    // 归一化 + BGR2RGB + HWC2CHW
    int dst_idx = dst_y * params.dst_width + dst_x;
    dst[0 * dst_area + dst_idx] = (r - params.mean[0]) / params.std[0];  // R
    dst[1 * dst_area + dst_idx] = (g - params.mean[1]) / params.std[1];  // G
    dst[2 * dst_area + dst_idx] = (b - params.mean[2]) / params.std[2];  // B
}

// 双线性插值预处理 kernel (质量好)
__global__ void preprocess_bilinear_kernel(
    const uint8_t* __restrict__ src,
    float* __restrict__ dst,
    PreprocessParams params
) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x >= params.dst_width || dst_y >= params.dst_height) return;
    
    int dst_area = params.dst_width * params.dst_height;
    
    float r, g, b;
    
    int x_in_resized = dst_x - params.pad_left;
    int y_in_resized = dst_y - params.pad_top;
    
    if (x_in_resized >= 0 && x_in_resized < params.new_width &&
        y_in_resized >= 0 && y_in_resized < params.new_height) {
        
        // 映射回原图坐标 (浮点)
        float src_x_f = x_in_resized / params.scale;
        float src_y_f = y_in_resized / params.scale;
        
        // 双线性插值的四个角点
        int x0 = (int)floorf(src_x_f);
        int y0 = (int)floorf(src_y_f);
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        
        // 边界处理
        x0 = max(0, min(x0, params.src_width - 1));
        x1 = max(0, min(x1, params.src_width - 1));
        y0 = max(0, min(y0, params.src_height - 1));
        y1 = max(0, min(y1, params.src_height - 1));
        
        // 插值权重
        float wx = src_x_f - floorf(src_x_f);
        float wy = src_y_f - floorf(src_y_f);
        
        // 读取四个角点的像素
        int idx00 = (y0 * params.src_width + x0) * 3;
        int idx01 = (y0 * params.src_width + x1) * 3;
        int idx10 = (y1 * params.src_width + x0) * 3;
        int idx11 = (y1 * params.src_width + x1) * 3;
        
        // 双线性插值 (BGR)
        b = (1-wx)*(1-wy)*src[idx00+0] + wx*(1-wy)*src[idx01+0] +
            (1-wx)*wy*src[idx10+0] + wx*wy*src[idx11+0];
        g = (1-wx)*(1-wy)*src[idx00+1] + wx*(1-wy)*src[idx01+1] +
            (1-wx)*wy*src[idx10+1] + wx*wy*src[idx11+1];
        r = (1-wx)*(1-wy)*src[idx00+2] + wx*(1-wy)*src[idx01+2] +
            (1-wx)*wy*src[idx10+2] + wx*wy*src[idx11+2];
    } else {
        r = g = b = 114.0f;
    }
    
    // 归一化 + BGR2RGB + HWC2CHW
    int dst_idx = dst_y * params.dst_width + dst_x;
    dst[0 * dst_area + dst_idx] = (r - params.mean[0]) / params.std[0];
    dst[1 * dst_area + dst_idx] = (g - params.mean[1]) / params.std[1];
    dst[2 * dst_area + dst_idx] = (b - params.mean[2]) / params.std[2];
}
