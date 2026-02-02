//
// TensorRT Engine 构建工具
// 支持 FP32 / FP16 / INT8 一键生成
//

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include "calibrator.h"

class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
};

enum class Precision {
    FP32,
    FP16,
    INT8
};

struct BuildConfig {
    std::string onnx_path;
    std::string output_dir;
    std::string model_name;
    std::string calib_data_dir;  // INT8 校准数据目录
    
    int input_h = 640;
    int input_w = 640;
    int min_batch = 1;
    int opt_batch = 1;
    int max_batch = 1;
    
    size_t workspace_size = 1ULL << 30;  // 1GB
    int calib_batch_size = 1;
    int max_calib_images = 100;
};

class EngineBuilder {
private:
    TRTLogger logger_;
    BuildConfig config_;

public:
    explicit EngineBuilder(const BuildConfig& config) : config_(config) {}

    bool build(Precision precision) {
        std::string precision_str;
        switch (precision) {
            case Precision::FP32: precision_str = "fp32"; break;
            case Precision::FP16: precision_str = "fp16"; break;
            case Precision::INT8: precision_str = "int8"; break;
        }

        std::string engine_path = config_.output_dir + "/" + config_.model_name + "_" + precision_str + ".engine";
        std::cout << "\n========================================\n";
        std::cout << "Building " << precision_str << " engine...\n";
        std::cout << "Output: " << engine_path << "\n";
        std::cout << "========================================\n";

        // 创建 Builder
        auto builder = std::unique_ptr<nvinfer1::IBuilder>(
            nvinfer1::createInferBuilder(logger_));
        if (!builder) {
            std::cerr << "Failed to create builder" << std::endl;
            return false;
        }

        // 创建 Network
        const auto explicitBatch = 1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
            builder->createNetworkV2(explicitBatch));
        if (!network) {
            std::cerr << "Failed to create network" << std::endl;
            return false;
        }

        // 解析 ONNX
        auto parser = std::unique_ptr<nvonnxparser::IParser>(
            nvonnxparser::createParser(*network, logger_));
        if (!parser->parseFromFile(config_.onnx_path.c_str(), 
                                   static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            std::cerr << "Failed to parse ONNX: " << config_.onnx_path << std::endl;
            return false;
        }
        std::cout << "ONNX parsed successfully" << std::endl;

        // 配置
        auto builderConfig = std::unique_ptr<nvinfer1::IBuilderConfig>(
            builder->createBuilderConfig());
        builderConfig->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 
                                          config_.workspace_size);

        // 设置 Optimization Profile (支持动态 batch)
        auto profile = builder->createOptimizationProfile();
        const char* input_name = network->getInput(0)->getName();
        
        profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN,
            nvinfer1::Dims4(config_.min_batch, 3, config_.input_h, config_.input_w));
        profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT,
            nvinfer1::Dims4(config_.opt_batch, 3, config_.input_h, config_.input_w));
        profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX,
            nvinfer1::Dims4(config_.max_batch, 3, config_.input_h, config_.input_w));
        
        builderConfig->addOptimizationProfile(profile);
        std::cout << "Batch range: [" << config_.min_batch << ", " 
                  << config_.opt_batch << ", " << config_.max_batch << "]" << std::endl;

        // 精度设置
        std::unique_ptr<Int8EntropyCalibrator> calibrator;

        switch (precision) {
            case Precision::FP32:
                // 默认就是 FP32，不需要额外设置
                break;

            case Precision::FP16:
                if (builder->platformHasFastFp16()) {
                    builderConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
                    std::cout << "FP16 enabled" << std::endl;
                } else {
                    std::cerr << "Warning: Platform does not support fast FP16" << std::endl;
                }
                break;

            case Precision::INT8:
                if (!builder->platformHasFastInt8()) {
                    std::cerr << "Error: Platform does not support INT8" << std::endl;
                    return false;
                }
                
                // INT8 通常也需要 FP16 支持
                if (builder->platformHasFastFp16()) {
                    builderConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
                }
                builderConfig->setFlag(nvinfer1::BuilderFlag::kINT8);

                // 创建校准器
                std::string calib_cache = config_.output_dir + "/" + config_.model_name + "_calib.cache";
                calibrator = std::make_unique<Int8EntropyCalibrator>(
                    config_.calib_batch_size,
                    config_.calib_data_dir,
                    calib_cache,
                    config_.input_h,
                    config_.input_w
                );
                builderConfig->setInt8Calibrator(calibrator.get());
                std::cout << "INT8 calibration enabled" << std::endl;
                std::cout << "Calibration data: " << config_.calib_data_dir << std::endl;
                break;
        }

        // 构建 Engine
        std::cout << "Building engine (this may take a while)..." << std::endl;
        auto plan = std::unique_ptr<nvinfer1::IHostMemory>(
            builder->buildSerializedNetwork(*network, *builderConfig));
        
        if (!plan) {
            std::cerr << "Failed to build engine" << std::endl;
            return false;
        }

        // 保存
        std::ofstream engine_file(engine_path, std::ios::binary);
        if (!engine_file) {
            std::cerr << "Failed to open output file: " << engine_path << std::endl;
            return false;
        }
        engine_file.write(reinterpret_cast<const char*>(plan->data()), plan->size());
        engine_file.close();

        std::cout << "Engine saved: " << engine_path << std::endl;
        std::cout << "Engine size: " << plan->size() / (1024.0 * 1024.0) << " MB" << std::endl;

        return true;
    }

    // 一键构建所有精度
    void build_all() {
        std::cout << "\n╔════════════════════════════════════════╗\n";
        std::cout << "║     Building All Precision Engines     ║\n";
        std::cout << "╚════════════════════════════════════════╝\n";

        bool fp32_ok = build(Precision::FP32);
        bool fp16_ok = build(Precision::FP16);
        bool int8_ok = build(Precision::INT8);

        std::cout << "\n========== Build Summary ==========\n";
        std::cout << "FP32: " << (fp32_ok ? "SUCCESS" : "FAILED") << "\n";
        std::cout << "FP16: " << (fp16_ok ? "SUCCESS" : "FAILED") << "\n";
        std::cout << "INT8: " << (int8_ok ? "SUCCESS" : "FAILED") << "\n";
    }
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --onnx <path>       ONNX model path (required)\n";
    std::cout << "  --output <dir>      Output directory (default: current dir)\n";
    std::cout << "  --name <name>       Model name for output files (default: model)\n";
    std::cout << "  --calib <dir>       Calibration images directory (required for INT8)\n";
    std::cout << "  --precision <p>     Precision: fp32, fp16, int8, all (default: all)\n";
    std::cout << "  --batch <n>         Optimal batch size (default: 1)\n";
    std::cout << "  --max-batch <n>     Maximum batch size (default: 1)\n";
    std::cout << "  --input-size <n>    Input size (default: 640)\n";
    std::cout << "\nExample:\n";
    std::cout << "  " << prog << " --onnx yolov5s.onnx --output ./engines --name yolov5s \\\n";
    std::cout << "              --calib ./coco_val --precision all --batch 8 --max-batch 16\n";
}

int main(int argc, char** argv) {
    BuildConfig config;
    config.output_dir = ".";
    config.model_name = "model";
    std::string precision = "all";

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--onnx" && i + 1 < argc) {
            config.onnx_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_dir = argv[++i];
        } else if (arg == "--name" && i + 1 < argc) {
            config.model_name = argv[++i];
        } else if (arg == "--calib" && i + 1 < argc) {
            config.calib_data_dir = argv[++i];
        } else if (arg == "--precision" && i + 1 < argc) {
            precision = argv[++i];
        } else if (arg == "--batch" && i + 1 < argc) {
            config.opt_batch = std::stoi(argv[++i]);
        } else if (arg == "--max-batch" && i + 1 < argc) {
            config.max_batch = std::stoi(argv[++i]);
        } else if (arg == "--input-size" && i + 1 < argc) {
            config.input_h = config.input_w = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    // 验证必要参数
    if (config.onnx_path.empty()) {
        std::cerr << "Error: --onnx is required\n\n";
        print_usage(argv[0]);
        return 1;
    }

    if ((precision == "int8" || precision == "all") && config.calib_data_dir.empty()) {
        std::cerr << "Error: --calib is required for INT8 precision\n\n";
        print_usage(argv[0]);
        return 1;
    }

    // 构建
    EngineBuilder builder(config);

    if (precision == "all") {
        builder.build_all();
    } else if (precision == "fp32") {
        builder.build(Precision::FP32);
    } else if (precision == "fp16") {
        builder.build(Precision::FP16);
    } else if (precision == "int8") {
        builder.build(Precision::INT8);
    } else {
        std::cerr << "Unknown precision: " << precision << std::endl;
        return 1;
    }

    return 0;
}
