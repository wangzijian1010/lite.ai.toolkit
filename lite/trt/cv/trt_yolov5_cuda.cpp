//
// YOLOv5 TensorRT 推理 - CUDA 预处理 + 后处理优化版本
//

#include "trt_yolov5_cuda.h"

using trtcv::TRTYoloV5CUDA;

TRTYoloV5CUDA::TRTYoloV5CUDA(const std::string &_trt_model_path, unsigned int _num_threads)
    : BasicTRTHandler(_trt_model_path, _num_threads)
{
    // 初始化 CUDA 预处理管理器
    const int input_height = input_node_dims.at(2);
    const int input_width = input_node_dims.at(3);
    
    // 使用双线性插值，质量更好
    preprocess_manager_ = std::make_unique<PreprocessManager>(input_width, input_height, true);
    preprocess_manager_->set_stream(stream);
    
    // 初始化 CUDA 后处理管理器
    postprocess_manager_ = std::make_unique<YoloV5PostprocessManager>(max_nms);
    postprocess_manager_->set_stream(stream);
}

void TRTYoloV5CUDA::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                        float iou_threshold, unsigned int topk, unsigned int nms_type)
{
    if (nms_type == NMS::BLEND) lite::utils::blending_nms(input, output, iou_threshold, topk);
    else if (nms_type == NMS::OFFSET) lite::utils::offset_nms(input, output, iou_threshold, topk);
    else lite::utils::hard_nms(input, output, iou_threshold, topk);
}

void TRTYoloV5CUDA::detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                            float score_threshold, float iou_threshold,
                            unsigned int topk, unsigned int nms_type)
{
    if (mat.empty()) return;

    int img_height = mat.rows;
    int img_width = mat.cols;

    // ============ 1. CUDA 预处理 (直接输出到 GPU) ============
    PreprocessResult preprocess_result = preprocess_manager_->preprocess(mat, static_cast<float*>(buffers[0]));

    // ============ 2. TensorRT 推理 ============
    bool status = trt_context->enqueueV3(stream);
    if (!status) {
        std::cerr << "Failed to infer by TensorRT." << std::endl;
        return;
    }

    // ============ 3. CUDA 后处理 (GPU 上过滤，只传有效数据) ============
    auto pred_dims = output_node_dims[0];
    const int num_anchors = pred_dims.at(1);
    const int num_classes = pred_dims.at(2) - 5;
    
    std::vector<FilteredBox> filtered_boxes = postprocess_manager_->process(
        static_cast<float*>(buffers[1]),
        num_anchors,
        num_classes,
        score_threshold,
        preprocess_result.scale,
        preprocess_result.pad_left,
        preprocess_result.pad_top,
        img_width,
        img_height
    );

#if LITETRT_DEBUG
    std::cout << "detected num_anchors: " << num_anchors << "\n";
    std::cout << "filtered boxes (before NMS): " << filtered_boxes.size() << "\n";
#endif

    // ============ 4. 转换为 Boxf 格式并做 NMS ============
    std::vector<types::Boxf> bbox_collection;
    bbox_collection.reserve(filtered_boxes.size());
    
    for (const auto& fb : filtered_boxes) {
        types::Boxf box;
        box.x1 = fb.x1;
        box.y1 = fb.y1;
        box.x2 = fb.x2;
        box.y2 = fb.y2;
        box.score = fb.score;
        box.label = fb.label;
        box.label_text = class_names[fb.label];
        box.flag = true;
        bbox_collection.push_back(box);
    }
    
    nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
}
