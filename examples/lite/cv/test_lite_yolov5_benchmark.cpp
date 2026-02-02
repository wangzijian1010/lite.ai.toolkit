//
// YOLOv5 性能基准测试
// 对比 原版 vs CUDA预处理 版本的速度
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
    // 默认路径
    std::string fp32_engine = "/workspace/lite.ai.toolkit/examples/hub/onnx/cv/yolov5s_fp32.engine";
    std::string fp16_engine = "/workspace/lite.ai.toolkit/examples/hub/onnx/cv/yolov5s_fp16.engine";
    std::string int8_engine = "/workspace/lite.ai.toolkit/examples/hub/onnx/cv/yolov5s_int8.engine";
    std::string test_img_path = "/workspace/lite.ai.toolkit/examples/logs/test_lite_yolov5_1.jpg";

    if (argc >= 5) {
        fp32_engine = argv[1];
        fp16_engine = argv[2];
        int8_engine = argv[3];
        test_img_path = argv[4];
    }

    cv::Mat test_img = cv::imread(test_img_path);
    if (test_img.empty()) {
        std::cerr << "Failed to load test image: " << test_img_path << std::endl;
        return -1;
    }
    std::cout << "Test image size: " << test_img.cols << "x" << test_img.rows << std::endl;

    const int warmup_runs = 50;
    const int benchmark_runs = 200;

    std::vector<BenchmarkResult> results;

    // ============ 原版 YOLOv5 (OpenCV 预处理) ============
    if (std::ifstream(fp16_engine).good()) {
        std::cout << "\n[Original] Loading engine: " << fp16_engine << std::endl;
        auto model = std::make_unique<lite::trt::cv::detection::YOLOV5>(fp16_engine);
        
        std::vector<lite::types::Boxf> boxes;
        std::vector<double> latencies;

        // Warmup
        std::cout << "[Original] Warming up..." << std::endl;
        for (int i = 0; i < warmup_runs; ++i) {
            boxes.clear();
            model->detect(test_img, boxes);
        }

        // Benchmark
        std::cout << "[Original] Benchmarking (" << benchmark_runs << " runs)..." << std::endl;
        for (int i = 0; i < benchmark_runs; ++i) {
            boxes.clear();
            auto start = std::chrono::high_resolution_clock::now();
            model->detect(test_img, boxes);
            auto end = std::chrono::high_resolution_clock::now();
            latencies.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }

        BenchmarkResult result;
        result.precision = "Original";
        result.avg_latency_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        result.min_latency_ms = *std::min_element(latencies.begin(), latencies.end());
        result.max_latency_ms = *std::max_element(latencies.begin(), latencies.end());
        result.throughput_fps = 1000.0 / result.avg_latency_ms;
        result.detected_boxes = static_cast<int>(boxes.size());
        results.push_back(result);
    }

    // ============ CUDA 预处理版本 ============
    if (std::ifstream(fp16_engine).good()) {
        std::cout << "\n[CUDA] Loading engine: " << fp16_engine << std::endl;
        auto model = std::make_unique<lite::trt::cv::detection::YOLOV5CUDA>(fp16_engine);
        
        std::vector<lite::types::Boxf> boxes;
        std::vector<double> latencies;

        // Warmup
        std::cout << "[CUDA] Warming up..." << std::endl;
        for (int i = 0; i < warmup_runs; ++i) {
            boxes.clear();
            model->detect(test_img, boxes);
        }

        // Benchmark
        std::cout << "[CUDA] Benchmarking (" << benchmark_runs << " runs)..." << std::endl;
        for (int i = 0; i < benchmark_runs; ++i) {
            boxes.clear();
            auto start = std::chrono::high_resolution_clock::now();
            model->detect(test_img, boxes);
            auto end = std::chrono::high_resolution_clock::now();
            latencies.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }

        BenchmarkResult result;
        result.precision = "CUDA";
        result.avg_latency_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        result.min_latency_ms = *std::min_element(latencies.begin(), latencies.end());
        result.max_latency_ms = *std::max_element(latencies.begin(), latencies.end());
        result.throughput_fps = 1000.0 / result.avg_latency_ms;
        result.detected_boxes = static_cast<int>(boxes.size());
        results.push_back(result);
    }

    // 打印结果
    if (!results.empty()) {
        YOLOv5Benchmark::print_results(results);
    }

    // 验证 CUDA 版本的检测结果正确性
    if (results.size() >= 2) {
        std::cout << "\n========== Accuracy Comparison ==========\n";
        
        auto original_model = std::make_unique<lite::trt::cv::detection::YOLOV5>(fp16_engine);
        std::vector<lite::types::Boxf> original_boxes;
        original_model->detect(test_img, original_boxes);

        auto cuda_model = std::make_unique<lite::trt::cv::detection::YOLOV5CUDA>(fp16_engine);
        std::vector<lite::types::Boxf> cuda_boxes;
        cuda_model->detect(test_img, cuda_boxes);

        compare_accuracy(original_boxes, cuda_boxes, "CUDA Preprocess");
    }

#else
    std::cout << "TensorRT not enabled. Please compile with -DENABLE_TENSORRT=ON" << std::endl;
#endif

    return 0;
}
