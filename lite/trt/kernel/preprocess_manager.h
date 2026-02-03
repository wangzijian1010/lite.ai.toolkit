//
// 预处理 CUDA Manager
// 提供 C++ 友好的接口，管理 GPU 内存
//

#ifndef LITE_AI_TOOLKIT_PREPROCESS_MANAGER_H
#define LITE_AI_TOOLKIT_PREPROCESS_MANAGER_H

#include "preprocess.cuh"
#include <opencv2/opencv.hpp>
#include <memory>
#include <cuda_runtime.h>

namespace trtcv {

// 预处理结果，包含缩放参数用于后处理还原坐标
struct PreprocessResult {
    float scale;        // 缩放比例
    int pad_left;       // 左边 padding
    int pad_top;        // 上边 padding
    int new_width;      // 缩放后宽度
    int new_height;     // 缩放后高度
    int src_width;      // 原图宽度
    int src_height;     // 原图高度
};

class PreprocessManager {
public:
    // 构造函数
    // target_width/height: 模型输入尺寸 (如 640x640)
    // use_bilinear: 是否使用双线性插值 (质量好但稍慢)
    PreprocessManager(int target_width, int target_height, bool use_bilinear = false);
    ~PreprocessManager();

    // 禁止拷贝
    PreprocessManager(const PreprocessManager&) = delete;
    PreprocessManager& operator=(const PreprocessManager&) = delete;

    // 预处理单张图片
    // 输入: cv::Mat BGR 图像
    // 输出: GPU 上的 float 张量指针 (CHW 格式)
    // 返回: 预处理参数 (用于后处理坐标还原)
    PreprocessResult preprocess(const cv::Mat& input, float* d_output);

    // 预处理并返回 GPU 内存 (内部管理)
    // 适合不想自己管理 GPU 内存的场景
    PreprocessResult preprocess(const cv::Mat& input);
    
    // 获取内部管理的输出 GPU 指针
    float* get_output_ptr() const { return d_output_; }

    // 预处理 (输入已经在 GPU 上)
    // 适合视频流场景，避免重复 H2D 拷贝
    PreprocessResult preprocess_gpu(const uint8_t* d_input, int src_width, int src_height, float* d_output);

    // 设置归一化参数 (默认 mean=0, std=255)
    void set_normalize_params(float mean[3], float std[3]);

    // 获取目标尺寸
    int get_target_width() const { return target_width_; }
    int get_target_height() const { return target_height_; }

    // 设置 CUDA stream (用于异步执行)
    void set_stream(cudaStream_t stream) { stream_ = stream; }

private:
    int target_width_;
    int target_height_;
    bool use_bilinear_;
    
    // GPU 内存
    uint8_t* d_input_ = nullptr;    // 输入图像 GPU 缓冲
    float* d_output_ = nullptr;     // 输出张量 GPU 缓冲
    size_t input_buffer_size_ = 0;  // 当前输入缓冲大小
    
    // Pinned Memory (加速 H2D 传输)
    uint8_t* h_pinned_input_ = nullptr;  // 锁页内存缓冲
    size_t pinned_buffer_size_ = 0;      // 锁页内存大小
    
    // 归一化参数
    float mean_[3] = {0.0f, 0.0f, 0.0f};
    float std_[3] = {255.0f, 255.0f, 255.0f};
    
    // CUDA stream
    cudaStream_t stream_ = nullptr;
    
    // 确保输入缓冲足够大
    void ensure_input_buffer(size_t size);
    
    // 确保 pinned memory 缓冲足够大
    void ensure_pinned_buffer(size_t size);
};

} // namespace trtcv

#endif //LITE_AI_TOOLKIT_PREPROCESS_MANAGER_H
