#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <algorithm>
#include <iterator>
#include <NvInfer.h>
#include <NvOnnxParser.h> // 【关键】之前报错就是缺这个
#include <opencv2/opencv.hpp>

// 引入你写好的校准器头文件
#include "calibrator.h"

// TensorRT 需要一个 Logger 来打印日志
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // 只打印警告和错误，避免信息太多
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

void build_engine(const std::string& onnx_path,
                  const std::string& engine_out_path,
                  bool use_int8,
                  const std::string& calib_imgs_dir = "") {

    // -----------------------------------------------------------
    // 第一步：创建 Builder 和 Network
    // -----------------------------------------------------------
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder) { std::cerr << "Create builder failed!" << std::endl; return; }

    // Explicit Batch 是现代 TRT 的标准模式 (1U << 0)
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) { std::cerr << "Create network failed!" << std::endl; return; }

    // -----------------------------------------------------------
    // 第二步：使用 Parser 解析 ONNX 文件
    // -----------------------------------------------------------
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
        std::cerr << "Parse ONNX failed!" << std::endl;
        return;
    }
    std::cout << "Successfully parsed ONNX file: " << onnx_path << std::endl;


    // 无法修复动态的维度 TRT无法修改为动态
    // // ===========================================================
    // // 【新增修复代码】 强制把 Batch 维度改为动态 (-1)
    // // ===========================================================
    // // 获取第一个输入张量 (通常就是 images)
    // auto input = network->getInput(0);
    //
    // // 获取当前维度 (比如 [1, 3, 640, 640])
    // auto dims = input->getDimensions();
    //
    // // 打印修改前的维度看看
    // std::cout << "Original Input Dims: " << dims.d[0] << "x" << dims.d[1] << "x" << dims.d[2] << "x" << dims.d[3] << std::endl;
    //
    // // 关键步骤：把第一个维度 (Batch) 改成 -1，表示动态
    // dims.d[0] = -1;
    // input->setDimensions(dims);
    //
    // std::cout << "===> Fixed Input Dims to Dynamic: -1x" << dims.d[1] << "x" << dims.d[2] << "x" << dims.d[3] << std::endl;

    // -----------------------------------------------------------
    // 第三步：配置构建参数 (Config) —— 最关键的一步
    // -----------------------------------------------------------
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());

    // 设置最大工作空间 (显存)，例如 2GB
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 31);

    // 启用 FP16 (通常 INT8 混合精度也需要 FP16 支持)
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }



    // ===========================================================
    // 【新增】关键修改：添加 Dynamic Batch 支持
    // ===========================================================
    // 1. 创建优化配置文件
    auto profile = builder->createOptimizationProfile();

    // 2. 获取网络的一个输入名称 (YOLOv5 通常叫 "images"，如果不确定可以用 parser->getNetwork()->getInput(0)->getName())
    const char* input_name = "images";

    // 3. 设置三档维度：
    // kMIN: 最小能跑多少？ (Batch = 1)
    // kOPT: 最常用的 Batch 是多少？ (TensorRT 会针对这个大小做极致优化，比如设为 8)
    // kMAX: 最大允许跑多少？ (比如 16，超过这个数就会报错)
    profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 3, 640, 640));
    profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(8, 3, 640, 640));
    profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(16, 3, 640, 640));

    // 4. 将配置添加到 config 中
    config->addOptimizationProfile(profile);

    std::cout << "===> Optimization Profile Added: Batch size range [1, 16], Opt = 8" << std::endl;



    // ================== INT8 核心逻辑 ==================
    // 使用 unique_ptr 管理校准器生命周期
    std::unique_ptr<Int8EntropyCalibrator> calibrator_ptr;

    if (use_int8) {
        if (!builder->platformHasFastInt8()) {
            std::cerr << "Warning: Current GPU does not support INT8, fallback to FP16/FP32." << std::endl;
        } else {
            std::cout << "===> Enabling INT8 Calibration..." << std::endl;
            config->setFlag(nvinfer1::BuilderFlag::kINT8);

            // 假设输入是 640x640，根据你的模型实际情况修改
            int input_h = 640;
            int input_w = 640;

            // 实例化校准器
            calibrator_ptr.reset(new Int8EntropyCalibrator(
                1, // Batch Size
                calib_imgs_dir,
                "yolov5_calib.table", // 生成的校准表文件名
                input_h, input_w
            ));

            // 设置校准器
            config->setInt8Calibrator(calibrator_ptr.get());
        }
    }
    // ===================================================

    // -----------------------------------------------------------
    // 第四步：构建并序列化 Engine
    // -----------------------------------------------------------
    std::cout << "Building engine... This may take a while." << std::endl;

    // buildSerializedNetwork 返回的是 IHostMemory，包含二进制数据
    auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!plan) {
        std::cerr << "Build engine failed!" << std::endl;
        return;
    }

    // -----------------------------------------------------------
    // 第五步：保存到磁盘
    // -----------------------------------------------------------
    std::ofstream engine_file(engine_out_path, std::ios::binary);
    engine_file.write(reinterpret_cast<const char*>(plan->data()), plan->size());
    engine_file.close();

    std::cout << "Engine build success! Saved to: " << engine_out_path << std::endl;
}

int main(int argc, char** argv) {
    // 1. 硬编码默认路径 (防止不传参数直接报错)
    std::string onnx_path = "/home/ubuntu/lite.ai.toolkit/examples/hub/onnx/cv/yolov5s.onnx";
    std::string engine_path = "/home/ubuntu/lite.ai.toolkit/examples/hub/onnx/cv/yolov5s_int8.engine";
    std::string img_dir = "/home/ubuntu/lite.ai.toolkit/examples/lite/resources/coco_data/val2017";

    // 2. 【强制开启 INT8】
    // 只要你运行这个程序，默认就是开启量化模式
    bool use_int8 = true;

    // 3. (可选) 依然保留命令行覆盖功能
    // 这样你既可以直接点 Run，也可以在终端里传不同的路径
    if (argc >= 4) {
        onnx_path = argv[1];
        engine_path = argv[2];
        img_dir = argv[3];
        // 既然传了图片路径，那肯定是要跑 INT8
        use_int8 = true;
    }

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "INT8 Mode: " << (use_int8 ? "ON" : "OFF") << std::endl;
    std::cout << "ONNX Path: " << onnx_path << std::endl;
    std::cout << "Image Dir: " << img_dir << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    build_engine(onnx_path, engine_path, use_int8, img_dir);
    return 0;
}