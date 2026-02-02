//
// YOLOv5 性能基准测试
// 对比 FP32 / FP16 / INT8 的速度和精度
//

#include "lite/lite.h"
#include <chrono>
#include <numeric>
#include <iomanip>

struct BenchmarkResult {
    std::string precision;
    double avg_latency_ms;
    double min_latency_ms;
    double max_latency_ms;
    double throughput_fps;
    int detected_boxes;
};

class YOLOv5Benchmark {
public:
    // 单次推理计时
    static double measure_single_inference(
        lite::trt::cv::detection::YOLOV5* model,
        const cv::Mat& img,
        std::vector<lite::types::Boxf>& boxes
    ) {
        boxes.clear();
        auto start = std::chrono::high_resolution_clock::now();
        model->detect(img, boxes);
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    // 运行完整 benchmark
    static BenchmarkResult run_benchmark(
        const std::string& engine_path,
        const std::string& precision_name,
        const cv::Mat& test_img,
        int warmup_runs = 10,
        int benchmark_runs = 100
    ) {
        BenchmarkResult result;
        result.precision = precision_name;

        // 加载模型
        std::cout << "\n[" << precision_name << "] Loading engine: " << engine_path << std::endl;
        auto model = std::make_unique<lite::trt::cv::detection::YOLOV5>(engine_path);

        std::vector<lite::types::Boxf> boxes;
        std::vector<double> latencies;

        // Warmup
        std::cout << "[" << precision_name << "] Warming up (" << warmup_runs << " runs)..." << std::endl;
        for (int i = 0; i < warmup_runs; ++i) {
            measure_single_inference(model.get(), test_img, boxes);
        }

        // Benchmark
        std::cout << "[" << precision_name << "] Benchmarking (" << benchmark_runs << " runs)..." << std::endl;
        for (int i = 0; i < benchmark_runs; ++i) {
            double latency = measure_single_inference(model.get(), test_img, boxes);
            latencies.push_back(latency);
        }

        // 统计结果
        result.avg_latency_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        result.min_latency_ms = *std::min_element(latencies.begin(), latencies.end());
        result.max_latency_ms = *std::max_element(latencies.begin(), latencies.end());
        result.throughput_fps = 1000.0 / result.avg_latency_ms;
        result.detected_boxes = static_cast<int>(boxes.size());

        return result;
    }

    // 打印结果表格
    static void print_results(const std::vector<BenchmarkResult>& results) {
        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                    YOLOv5 Benchmark Results                      ║\n";
        std::cout << "╠══════════╦══════════╦══════════╦══════════╦══════════╦══════════╣\n";
        std::cout << "║ Precision║ Avg (ms) ║ Min (ms) ║ Max (ms) ║   FPS    ║  Boxes   ║\n";
        std::cout << "╠══════════╬══════════╬══════════╬══════════╬══════════╬══════════╣\n";

        for (const auto& r : results) {
            std::cout << "║ " << std::setw(8) << r.precision << " ║"
                      << std::setw(9) << std::fixed << std::setprecision(2) << r.avg_latency_ms << " ║"
                      << std::setw(9) << r.min_latency_ms << " ║"
                      << std::setw(9) << r.max_latency_ms << " ║"
                      << std::setw(9) << r.throughput_fps << " ║"
                      << std::setw(9) << r.detected_boxes << " ║\n";
        }

        std::cout << "╚══════════╩══════════╩══════════╩══════════╩══════════╩══════════╝\n";

        // 计算加速比
        if (results.size() > 1) {
            double baseline = results[0].avg_latency_ms;
            std::cout << "\nSpeedup vs " << results[0].precision << ":\n";
            for (size_t i = 1; i < results.size(); ++i) {
                double speedup = baseline / results[i].avg_latency_ms;
                std::cout << "  " << results[i].precision << ": " 
                          << std::fixed << std::setprecision(2) << speedup << "x\n";
            }
        }
    }
};

// 精度对比：计算检测框的 IoU
float compute_iou(const lite::types::Boxf& a, const lite::types::Boxf& b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);

    float inter_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float a_area = (a.x2 - a.x1) * (a.y2 - a.y1);
    float b_area = (b.x2 - b.x1) * (b.y2 - b.y1);

    return inter_area / (a_area + b_area - inter_area + 1e-6f);
}

void compare_accuracy(
    const std::vector<lite::types::Boxf>& baseline_boxes,
    const std::vector<lite::types::Boxf>& test_boxes,
    const std::string& test_name
) {
    std::cout << "\n[Accuracy] " << test_name << " vs Baseline:\n";
    std::cout << "  Baseline boxes: " << baseline_boxes.size() << "\n";
    std::cout << "  Test boxes: " << test_boxes.size() << "\n";

    // 计算匹配的框数量 (IoU > 0.5)
    int matched = 0;
    for (const auto& test_box : test_boxes) {
        for (const auto& base_box : baseline_boxes) {
            if (test_box.label == base_box.label && compute_iou(test_box, base_box) > 0.5f) {
                matched++;
                break;
            }
        }
    }

    float precision = test_boxes.empty() ? 0 : (float)matched / test_boxes.size();
    float recall = baseline_boxes.empty() ? 0 : (float)matched / baseline_boxes.size();

    std::cout << "  Matched: " << matched << "\n";
    std::cout << "  Precision: " << std::fixed << std::setprecision(2) << precision * 100 << "%\n";
    std::cout << "  Recall: " << std::fixed << std::setprecision(2) << recall * 100 << "%\n";
}

int main(int argc, char* argv[]) {
#ifdef ENABLE_TENSORRT
    // 默认路径 - 根据你的实际路径修改
    std::string fp32_engine = "/workspace/lite.ai.toolkit/examples/hub/onnx/cv/yolov5s_fp32.engine";
    std::string fp16_engine = "/workspace/lite.ai.toolkit/examples/hub/onnx/cv/yolov5s_fp16.engine";
    std::string int8_engine = "/workspace/lite.ai.toolkit/examples/hub/onnx/cv/yolov5s_int8.engine";
    std::string test_img_path = "/workspace/lite.ai.toolkit/examples/logs/test_lite_yolov5_1.jpg";

    // 命令行参数覆盖
    if (argc >= 5) {
        fp32_engine = argv[1];
        fp16_engine = argv[2];
        int8_engine = argv[3];
        test_img_path = argv[4];
    }

    // 加载测试图片
    cv::Mat test_img = cv::imread(test_img_path);
    if (test_img.empty()) {
        std::cerr << "Failed to load test image: " << test_img_path << std::endl;
        return -1;
    }
    std::cout << "Test image size: " << test_img.cols << "x" << test_img.rows << std::endl;

    std::vector<BenchmarkResult> results;

    // 测试 FP32
    if (std::ifstream(fp32_engine).good()) {
        results.push_back(YOLOv5Benchmark::run_benchmark(fp32_engine, "FP32", test_img));
    } else {
        std::cout << "[SKIP] FP32 engine not found: " << fp32_engine << std::endl;
    }

    // 测试 FP16
    if (std::ifstream(fp16_engine).good()) {
        results.push_back(YOLOv5Benchmark::run_benchmark(fp16_engine, "FP16", test_img));
    } else {
        std::cout << "[SKIP] FP16 engine not found: " << fp16_engine << std::endl;
    }

    // 测试 INT8
    if (std::ifstream(int8_engine).good()) {
        results.push_back(YOLOv5Benchmark::run_benchmark(int8_engine, "INT8", test_img));
    } else {
        std::cout << "[SKIP] INT8 engine not found: " << int8_engine << std::endl;
    }

    // 打印结果
    if (!results.empty()) {
        YOLOv5Benchmark::print_results(results);
    }

    // 精度对比 (如果有多个精度的 engine)
    if (results.size() >= 2) {
        std::cout << "\n========== Accuracy Comparison ==========\n";
        
        // 用 FP32 作为 baseline
        auto fp32_model = std::make_unique<lite::trt::cv::detection::YOLOV5>(fp32_engine);
        std::vector<lite::types::Boxf> fp32_boxes;
        fp32_model->detect(test_img, fp32_boxes);

        if (std::ifstream(int8_engine).good()) {
            auto int8_model = std::make_unique<lite::trt::cv::detection::YOLOV5>(int8_engine);
            std::vector<lite::types::Boxf> int8_boxes;
            int8_model->detect(test_img, int8_boxes);
            compare_accuracy(fp32_boxes, int8_boxes, "INT8");
        }
    }

#else
    std::cout << "TensorRT not enabled. Please compile with -DENABLE_TENSORRT=ON" << std::endl;
#endif

    return 0;
}
