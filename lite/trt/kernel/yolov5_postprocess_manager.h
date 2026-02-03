//
// YOLOv5 后处理 CUDA Manager
// 管理 GPU 内存，提供 C++ 友好接口
//

#ifndef LITE_AI_TOOLKIT_YOLOV5_POSTPROCESS_MANAGER_H
#define LITE_AI_TOOLKIT_YOLOV5_POSTPROCESS_MANAGER_H

#include "yolov5_postprocess.cuh"
#include <vector>
#include <cuda_runtime.h>

namespace trtcv {

class YoloV5PostprocessManager {
public:
    // max_boxes: 最大输出 bbox 数量（防止显存溢出）
    explicit YoloV5PostprocessManager(int max_boxes = 1000);
    ~YoloV5PostprocessManager();

    // 禁止拷贝
    YoloV5PostprocessManager(const YoloV5PostprocessManager&) = delete;
    YoloV5PostprocessManager& operator=(const YoloV5PostprocessManager&) = delete;

    // 执行后处理
    // predictions: GPU 上的 TensorRT 输出
    // 返回: 过滤后的 bbox 列表（已拷贝到 CPU）
    std::vector<FilteredBox> process(
        const float* d_predictions,
        int num_anchors,
        int num_classes,
        float score_threshold,
        float scale,
        int pad_left,
        int pad_top,
        int src_width,
        int src_height
    );

    // 设置 CUDA stream
    void set_stream(cudaStream_t stream) { stream_ = stream; }

private:
    int max_boxes_;
    
    // GPU 内存
    FilteredBox* d_output_boxes_ = nullptr;
    int* d_output_count_ = nullptr;
    
    // CPU 端 pinned memory（加速 D2H）
    int* h_output_count_ = nullptr;
    FilteredBox* h_output_boxes_ = nullptr;
    
    cudaStream_t stream_ = nullptr;
};

} // namespace trtcv

#endif // LITE_AI_TOOLKIT_YOLOV5_POSTPROCESS_MANAGER_H
