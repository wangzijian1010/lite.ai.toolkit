//
// 预处理 CUDA Manager 实现
//

#include "preprocess_manager.h"
#include <iostream>
#include <stdexcept>

namespace trtcv {

PreprocessManager::PreprocessManager(int target_width, int target_height, bool use_bilinear)
    : target_width_(target_width)
    , target_height_(target_height)
    , use_bilinear_(use_bilinear)
{
    // 预分配输出缓冲 (固定大小)
    size_t output_size = target_width_ * target_height_ * 3 * sizeof(float);
    cudaError_t err = cudaMalloc(&d_output_, output_size);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for output");
    }
}

PreprocessManager::~PreprocessManager() {
    if (d_input_) cudaFree(d_input_);
    if (d_output_) cudaFree(d_output_);
}

void PreprocessManager::ensure_input_buffer(size_t size) {
    if (size > input_buffer_size_) {
        if (d_input_) cudaFree(d_input_);
        cudaError_t err = cudaMalloc(&d_input_, size);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate GPU memory for input");
        }
        input_buffer_size_ = size;
    }
}

void PreprocessManager::set_normalize_params(float mean[3], float std[3]) {
    for (int i = 0; i < 3; ++i) {
        mean_[i] = mean[i];
        std_[i] = std[i];
    }
}

PreprocessResult PreprocessManager::preprocess(const cv::Mat& input, float* d_output) {
    if (input.empty()) {
        throw std::runtime_error("Input image is empty");
    }
    if (input.type() != CV_8UC3) {
        throw std::runtime_error("Input image must be CV_8UC3 (BGR)");
    }

    int src_width = input.cols;
    int src_height = input.rows;
    size_t input_size = src_width * src_height * 3 * sizeof(uint8_t);

    // 确保输入缓冲足够大
    ensure_input_buffer(input_size);

    // H2D 拷贝
    if (stream_) {
        cudaMemcpyAsync(d_input_, input.data, input_size, cudaMemcpyHostToDevice, stream_);
    } else {
        cudaMemcpy(d_input_, input.data, input_size, cudaMemcpyHostToDevice);
    }

    // 调用 GPU 预处理
    return preprocess_gpu(d_input_, src_width, src_height, d_output);
}

PreprocessResult PreprocessManager::preprocess(const cv::Mat& input) {
    return preprocess(input, d_output_);
}

PreprocessResult PreprocessManager::preprocess_gpu(
    const uint8_t* d_input, 
    int src_width, 
    int src_height, 
    float* d_output
) {
    // 计算 letterbox 参数
    PreprocessParams params;
    compute_letterbox_params(src_width, src_height, target_width_, target_height_, params);
    
    // 设置归一化参数
    params.mean[0] = mean_[0];
    params.mean[1] = mean_[1];
    params.mean[2] = mean_[2];
    params.std[0] = std_[0];
    params.std[1] = std_[1];
    params.std[2] = std_[2];

    // 启动 kernel
    dim3 block(32, 32);
    dim3 grid(
        (target_width_ + block.x - 1) / block.x,
        (target_height_ + block.y - 1) / block.y
    );

    if (use_bilinear_) {
        if (stream_) {
            preprocess_bilinear_kernel<<<grid, block, 0, stream_>>>(d_input, d_output, params);
        } else {
            preprocess_bilinear_kernel<<<grid, block>>>(d_input, d_output, params);
        }
    } else {
        if (stream_) {
            preprocess_kernel<<<grid, block, 0, stream_>>>(d_input, d_output, params);
        } else {
            preprocess_kernel<<<grid, block>>>(d_input, d_output, params);
        }
    }

    // 检查 kernel 执行错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel error: ") + cudaGetErrorString(err));
    }

    // 返回预处理参数
    PreprocessResult result;
    result.scale = params.scale;
    result.pad_left = params.pad_left;
    result.pad_top = params.pad_top;
    result.new_width = params.new_width;
    result.new_height = params.new_height;
    result.src_width = src_width;
    result.src_height = src_height;

    return result;
}

} // namespace trtcv
