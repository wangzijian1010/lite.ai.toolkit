//
// YOLOv5 后处理 CUDA Kernel
// 在 GPU 上做置信度过滤，减少 D2H 传输量
//

#ifndef LITE_AI_TOOLKIT_YOLOV5_POSTPROCESS_CUH
#define LITE_AI_TOOLKIT_YOLOV5_POSTPROCESS_CUH

#include <cuda_runtime.h>

namespace trtcv {

// 过滤后的 bbox 结构（紧凑格式，便于传输）
struct FilteredBox {
    float x1, y1, x2, y2;  // 原图坐标
    float score;           // 最终置信度
    int label;             // 类别索引
};

// YOLOv5 后处理参数
struct YoloV5PostprocessParams {
    int num_anchors;       // anchor 数量 (如 25200)
    int num_classes;       // 类别数 (如 80)
    float score_threshold; // 置信度阈值
    float scale;           // letterbox 缩放比例
    int pad_left;          // 左边 padding
    int pad_top;           // 上边 padding
    int src_width;         // 原图宽度
    int src_height;        // 原图高度
};

// 后处理 kernel 声明
__global__ void yolov5_filter_boxes_kernel(
    const float* __restrict__ predictions,  // TensorRT 输出 [1, num_anchors, 85]
    FilteredBox* __restrict__ output_boxes, // 过滤后的 bbox
    int* __restrict__ output_count,         // 有效 bbox 数量
    YoloV5PostprocessParams params
);

// 辅助函数：重置计数器
__global__ void reset_counter_kernel(int* counter);

} // namespace trtcv

#endif // LITE_AI_TOOLKIT_YOLOV5_POSTPROCESS_CUH
