//
// YOLOv5 TensorRT 推理 - CUDA 预处理优化版本
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
    
    // 设置使用同一个 CUDA stream
    preprocess_manager_->set_stream(stream);
}

void TRTYoloV5CUDA::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                        float iou_threshold, unsigned int topk, unsigned int nms_type)
{
    if (nms_type == NMS::BLEND) lite::utils::blending_nms(input, output, iou_threshold, topk);
    else if (nms_type == NMS::OFFSET) lite::utils::offset_nms(input, output, iou_threshold, topk);
    else lite::utils::hard_nms(input, output, iou_threshold, topk);
}

void TRTYoloV5CUDA::generate_bboxes(const PreprocessResult &preprocess_result,
                                     std::vector<types::Boxf> &bbox_collection,
                                     float* output,
                                     float score_threshold,
                                     int img_height, int img_width)
{
    auto pred_dims = output_node_dims[0];
    const unsigned int num_anchors = pred_dims.at(1);
    const unsigned int num_classes = pred_dims.at(2) - 5;

    // 从预处理结果获取缩放参数
    float r_ = preprocess_result.scale;
    int dw_ = preprocess_result.pad_left;
    int dh_ = preprocess_result.pad_top;

    bbox_collection.clear();
    unsigned int count = 0;
    
    for (unsigned int i = 0; i < num_anchors; ++i)
    {
        float obj_conf = output[i * pred_dims.at(2) + 4];
        if (obj_conf < score_threshold) continue;

        float cls_conf = output[i * pred_dims.at(2) + 5];
        unsigned int label = 0;
        for (unsigned int j = 0; j < num_classes; ++j)
        {
            float tmp_conf = output[i * pred_dims.at(2) + 5 + j];
            if (tmp_conf > cls_conf)
            {
                cls_conf = tmp_conf;
                label = j;
            }
        }
        
        float conf = obj_conf * cls_conf;
        if (conf < score_threshold) continue;

        float cx = output[i * pred_dims.at(2)];
        float cy = output[i * pred_dims.at(2) + 1];
        float w = output[i * pred_dims.at(2) + 2];
        float h = output[i * pred_dims.at(2) + 3];
        
        // 还原到原图坐标
        float x1 = ((cx - w / 2.f) - (float)dw_) / r_;
        float y1 = ((cy - h / 2.f) - (float)dh_) / r_;
        float x2 = ((cx + w / 2.f) - (float)dw_) / r_;
        float y2 = ((cy + h / 2.f) - (float)dh_) / r_;

        types::Boxf box;
        box.x1 = std::max(0.f, x1);
        box.y1 = std::max(0.f, y1);
        box.x2 = std::min(x2, (float)img_width - 1.f);
        box.y2 = std::min(y2, (float)img_height - 1.f);
        box.score = conf;
        box.label = label;
        box.label_text = class_names[label];
        box.flag = true;
        bbox_collection.push_back(box);

        count += 1;
        if (count > max_nms) break;
    }

#if LITETRT_DEBUG
    std::cout << "detected num_anchors: " << num_anchors << "\n";
    std::cout << "generate_bboxes num: " << bbox_collection.size() << "\n";
#endif
}

void TRTYoloV5CUDA::detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                            float score_threshold, float iou_threshold,
                            unsigned int topk, unsigned int nms_type)
{
    if (mat.empty()) return;

    const int input_height = input_node_dims.at(2);
    const int input_width = input_node_dims.at(3);
    int img_height = static_cast<int>(mat.rows);
    int img_width = static_cast<int>(mat.cols);

    // ============ 1. CUDA 预处理 (直接输出到 GPU) ============
    // 预处理结果直接写入 buffers[0]，省掉一次 H2D 拷贝
    PreprocessResult preprocess_result = preprocess_manager_->preprocess(mat, static_cast<float*>(buffers[0]));

    // ============ 2. TensorRT 推理 ============
    bool status = trt_context->enqueueV3(stream);
    if (!status) {
        std::cerr << "Failed to infer by TensorRT." << std::endl;
        return;
    }

    // ============ 3. 拷贝输出到 CPU ============
    auto pred_dims = output_node_dims[0];
    std::vector<float> output(pred_dims[0] * pred_dims[1] * pred_dims[2]);
    
    cudaMemcpyAsync(output.data(), buffers[1], 
                    pred_dims[0] * pred_dims[1] * pred_dims[2] * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // ============ 4. 后处理 ============
    std::vector<types::Boxf> bbox_collection;
    generate_bboxes(preprocess_result, bbox_collection, output.data(), 
                    score_threshold, img_height, img_width);
    nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
}
