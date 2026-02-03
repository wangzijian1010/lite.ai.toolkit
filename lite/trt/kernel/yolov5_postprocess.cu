//
// YOLOv5 后处理 CUDA Kernel 实现
//

#include "yolov5_postprocess.cuh"

namespace trtcv {

__global__ void reset_counter_kernel(int* counter) {
    *counter = 0;
}

__global__ void yolov5_filter_boxes_kernel(
    const float* __restrict__ predictions,
    FilteredBox* __restrict__ output_boxes,
    int* __restrict__ output_count,
    YoloV5PostprocessParams params
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.num_anchors) return;

    // YOLOv5 输出格式: [cx, cy, w, h, obj_conf, cls_conf_0, cls_conf_1, ..., cls_conf_79]
    // 每个 anchor 有 (5 + num_classes) 个值
    const int stride = 5 + params.num_classes;
    const float* anchor_data = predictions + idx * stride;

    // 1. 先检查 objectness 置信度
    float obj_conf = anchor_data[4];
    if (obj_conf < params.score_threshold) return;

    // 2. 找最大类别置信度
    float max_cls_conf = 0.0f;
    int max_cls_idx = 0;
    
    for (int c = 0; c < params.num_classes; ++c) {
        float cls_conf = anchor_data[5 + c];
        if (cls_conf > max_cls_conf) {
            max_cls_conf = cls_conf;
            max_cls_idx = c;
        }
    }

    // 3. 计算最终置信度
    float final_score = obj_conf * max_cls_conf;
    if (final_score < params.score_threshold) return;

    // 4. 解码 bbox 坐标 (从 letterbox 坐标还原到原图坐标)
    float cx = anchor_data[0];
    float cy = anchor_data[1];
    float w = anchor_data[2];
    float h = anchor_data[3];

    // letterbox 逆变换
    float x1 = ((cx - w * 0.5f) - params.pad_left) / params.scale;
    float y1 = ((cy - h * 0.5f) - params.pad_top) / params.scale;
    float x2 = ((cx + w * 0.5f) - params.pad_left) / params.scale;
    float y2 = ((cy + h * 0.5f) - params.pad_top) / params.scale;

    // 裁剪到原图范围
    x1 = fmaxf(0.0f, fminf(x1, (float)(params.src_width - 1)));
    y1 = fmaxf(0.0f, fminf(y1, (float)(params.src_height - 1)));
    x2 = fmaxf(0.0f, fminf(x2, (float)(params.src_width - 1)));
    y2 = fmaxf(0.0f, fminf(y2, (float)(params.src_height - 1)));

    // 5. 原子操作获取输出位置
    int out_idx = atomicAdd(output_count, 1);

    // 6. 写入结果
    output_boxes[out_idx].x1 = x1;
    output_boxes[out_idx].y1 = y1;
    output_boxes[out_idx].x2 = x2;
    output_boxes[out_idx].y2 = y2;
    output_boxes[out_idx].score = final_score;
    output_boxes[out_idx].label = max_cls_idx;
}

} // namespace trtcv
