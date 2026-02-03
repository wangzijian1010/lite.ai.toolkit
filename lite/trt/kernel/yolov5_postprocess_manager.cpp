//
// YOLOv5 后处理 CUDA Manager 实现
//

#include "yolov5_postprocess_manager.h"
#include <stdexcept>
#include <algorithm>

namespace trtcv {

YoloV5PostprocessManager::YoloV5PostprocessManager(int max_boxes)
    : max_boxes_(max_boxes)
{
    // 分配 GPU 内存
    cudaError_t err;
    
    err = cudaMalloc(&d_output_boxes_, max_boxes_ * sizeof(FilteredBox));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for output boxes");
    }
    
    err = cudaMalloc(&d_output_count_, sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_output_boxes_);
        throw std::runtime_error("Failed to allocate GPU memory for output count");
    }
    
    // 分配 pinned memory（加速 D2H 传输）
    err = cudaMallocHost(&h_output_count_, sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_output_boxes_);
        cudaFree(d_output_count_);
        throw std::runtime_error("Failed to allocate pinned memory for output count");
    }
    
    err = cudaMallocHost(&h_output_boxes_, max_boxes_ * sizeof(FilteredBox));
    if (err != cudaSuccess) {
        cudaFree(d_output_boxes_);
        cudaFree(d_output_count_);
        cudaFreeHost(h_output_count_);
        throw std::runtime_error("Failed to allocate pinned memory for output boxes");
    }
}

YoloV5PostprocessManager::~YoloV5PostprocessManager() {
    if (d_output_boxes_) cudaFree(d_output_boxes_);
    if (d_output_count_) cudaFree(d_output_count_);
    if (h_output_count_) cudaFreeHost(h_output_count_);
    if (h_output_boxes_) cudaFreeHost(h_output_boxes_);
}

std::vector<FilteredBox> YoloV5PostprocessManager::process(
    const float* d_predictions,
    int num_anchors,
    int num_classes,
    float score_threshold,
    float scale,
    int pad_left,
    int pad_top,
    int src_width,
    int src_height
) {
    // 1. 重置计数器
    if (stream_) {
        reset_counter_kernel<<<1, 1, 0, stream_>>>(d_output_count_);
    } else {
        reset_counter_kernel<<<1, 1>>>(d_output_count_);
    }

    // 2. 准备参数
    YoloV5PostprocessParams params;
    params.num_anchors = num_anchors;
    params.num_classes = num_classes;
    params.score_threshold = score_threshold;
    params.scale = scale;
    params.pad_left = pad_left;
    params.pad_top = pad_top;
    params.src_width = src_width;
    params.src_height = src_height;

    // 3. 启动 kernel
    const int block_size = 256;
    const int grid_size = (num_anchors + block_size - 1) / block_size;
    
    if (stream_) {
        yolov5_filter_boxes_kernel<<<grid_size, block_size, 0, stream_>>>(
            d_predictions, d_output_boxes_, d_output_count_, params);
    } else {
        yolov5_filter_boxes_kernel<<<grid_size, block_size>>>(
            d_predictions, d_output_boxes_, d_output_count_, params);
    }

    // 4. 拷贝计数器到 CPU
    if (stream_) {
        cudaMemcpyAsync(h_output_count_, d_output_count_, sizeof(int), 
                        cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
    } else {
        cudaMemcpy(h_output_count_, d_output_count_, sizeof(int), cudaMemcpyDeviceToHost);
    }

    int count = std::min(*h_output_count_, max_boxes_);
    if (count == 0) {
        return {};
    }

    // 5. 只拷贝有效的 bbox（这是关键优化点！）
    if (stream_) {
        cudaMemcpyAsync(h_output_boxes_, d_output_boxes_, 
                        count * sizeof(FilteredBox), cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
    } else {
        cudaMemcpy(h_output_boxes_, d_output_boxes_, 
                   count * sizeof(FilteredBox), cudaMemcpyDeviceToHost);
    }

    // 6. 返回结果
    return std::vector<FilteredBox>(h_output_boxes_, h_output_boxes_ + count);
}

} // namespace trtcv
