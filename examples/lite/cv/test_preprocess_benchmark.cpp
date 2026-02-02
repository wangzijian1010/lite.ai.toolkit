//
// 预处理性能对比测试
// OpenCV CPU vs CUDA Kernel
//

#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "lite/trt/kernel/preprocess_manager.h"

// OpenCV CPU 预处理 (当前项目使用的方式)
void preprocess_opencv(const cv::Mat& input, std::vector<float>& output, int target_size) {
    int img_height = input.rows;
    int img_width = input.cols;
    
    // 1. 计算缩放比例
    float scale = std::min((float)target_size / img_width, (float)target_size / img_height);
    int new_width = (int)(img_width * scale);
    int new_height = (int)(img_height * scale);
    int pad_left = (target_size - new_width) / 2;
    int pad_top = (target_size - new_height) / 2;
    
    // 2. Resize
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(new_width, new_height));
    
    // 3. Letterbox (padding)
    cv::Mat letterboxed(target_size, target_size, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(letterboxed(cv::Rect(pad_left, pad_top, new_width, new_height)));
    
    // 4. BGR2RGB
    cv::Mat rgb;
    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
    
    // 5. Normalize + HWC2CHW
    cv::Mat normalized;
    rgb.convertTo(normalized, CV_32F, 1.0 / 255.0);
    
    // 6. HWC to CHW
    output.resize(3 * target_size * target_size);
    std::vector<cv::Mat> channels(3);
    cv::split(normalized, channels);
    
    int area = target_size * target_size;
    memcpy(output.data() + 0 * area, channels[0].data, area * sizeof(float));
    memcpy(output.data() + 1 * area, channels[1].data, area * sizeof(float));
    memcpy(output.data() + 2 * area, channels[2].data, area * sizeof(float));
}

struct BenchmarkResult {
    std::string name;
    double avg_ms;
    double min_ms;
    double max_ms;
};

void print_results(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║           Preprocess Benchmark Results                 ║\n";
    std::cout << "╠════════════════════╦══════════╦══════════╦════════════╣\n";
    std::cout << "║ Method             ║ Avg (ms) ║ Min (ms) ║ Speedup    ║\n";
    std::cout << "╠════════════════════╬══════════╬══════════╬════════════╣\n";
    
    double baseline = results[0].avg_ms;
    for (const auto& r : results) {
        double speedup = baseline / r.avg_ms;
        std::cout << "║ " << std::setw(18) << std::left << r.name << " ║"
                  << std::setw(9) << std::right << std::fixed << std::setprecision(3) << r.avg_ms << " ║"
                  << std::setw(9) << r.min_ms << " ║"
                  << std::setw(9) << std::setprecision(2) << speedup << "x ║\n";
    }
    std::cout << "╚════════════════════╩══════════╩══════════╩════════════╝\n";
}

int main(int argc, char* argv[]) {
    std::string img_path = "../../../examples/lite/resources/test_lite_yolov5_1.jpg";
    if (argc > 1) img_path = argv[1];
    
    cv::Mat input = cv::imread(img_path);
    if (input.empty()) {
        std::cerr << "Failed to load image: " << img_path << std::endl;
        return -1;
    }
    
    std::cout << "Input image: " << input.cols << "x" << input.rows << std::endl;
    
    const int target_size = 640;
    const int warmup = 50;
    const int iterations = 500;
    
    std::vector<BenchmarkResult> results;
    
    // ============ OpenCV CPU ============
    {
        std::vector<float> output;
        std::vector<double> times;
        
        // Warmup
        for (int i = 0; i < warmup; ++i) {
            preprocess_opencv(input, output, target_size);
        }
        
        // Benchmark
        for (int i = 0; i < iterations; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            preprocess_opencv(input, output, target_size);
            auto end = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }
        
        double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        double min_t = *std::min_element(times.begin(), times.end());
        double max_t = *std::max_element(times.begin(), times.end());
        
        results.push_back({"OpenCV CPU", avg, min_t, max_t});
    }
    
    // ============ CUDA Nearest ============
    {
        trtcv::PreprocessManager manager(target_size, target_size, false);
        std::vector<double> times;
        
        // Warmup
        for (int i = 0; i < warmup; ++i) {
            manager.preprocess(input);
            cudaDeviceSynchronize();
        }
        
        // Benchmark
        for (int i = 0; i < iterations; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            manager.preprocess(input);
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }
        
        double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        double min_t = *std::min_element(times.begin(), times.end());
        double max_t = *std::max_element(times.begin(), times.end());
        
        results.push_back({"CUDA Nearest", avg, min_t, max_t});
    }
    
    // ============ CUDA Bilinear ============
    {
        trtcv::PreprocessManager manager(target_size, target_size, true);
        std::vector<double> times;
        
        // Warmup
        for (int i = 0; i < warmup; ++i) {
            manager.preprocess(input);
            cudaDeviceSynchronize();
        }
        
        // Benchmark
        for (int i = 0; i < iterations; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            manager.preprocess(input);
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }
        
        double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        double min_t = *std::min_element(times.begin(), times.end());
        double max_t = *std::max_element(times.begin(), times.end());
        
        results.push_back({"CUDA Bilinear", avg, min_t, max_t});
    }
    
    // ============ CUDA (数据已在 GPU) ============
    {
        trtcv::PreprocessManager manager(target_size, target_size, false);
        
        // 预先把数据拷贝到 GPU
        uint8_t* d_input;
        size_t input_size = input.cols * input.rows * 3;
        cudaMalloc(&d_input, input_size);
        cudaMemcpy(d_input, input.data, input_size, cudaMemcpyHostToDevice);
        
        float* d_output;
        cudaMalloc(&d_output, target_size * target_size * 3 * sizeof(float));
        
        std::vector<double> times;
        
        // Warmup
        for (int i = 0; i < warmup; ++i) {
            manager.preprocess_gpu(d_input, input.cols, input.rows, d_output);
            cudaDeviceSynchronize();
        }
        
        // Benchmark (纯 GPU 时间，不含 H2D)
        for (int i = 0; i < iterations; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            manager.preprocess_gpu(d_input, input.cols, input.rows, d_output);
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }
        
        cudaFree(d_input);
        cudaFree(d_output);
        
        double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        double min_t = *std::min_element(times.begin(), times.end());
        double max_t = *std::max_element(times.begin(), times.end());
        
        results.push_back({"CUDA (GPU only)", avg, min_t, max_t});
    }
    
    print_results(results);
    
    // 验证结果正确性
    std::cout << "\n[Validation] Comparing OpenCV vs CUDA output...\n";
    {
        std::vector<float> opencv_output;
        preprocess_opencv(input, opencv_output, target_size);
        
        trtcv::PreprocessManager manager(target_size, target_size, true);
        manager.preprocess(input);
        cudaDeviceSynchronize();
        
        std::vector<float> cuda_output(3 * target_size * target_size);
        cudaMemcpy(cuda_output.data(), manager.get_output_ptr(), 
                   cuda_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
        
        // 计算差异
        double max_diff = 0;
        double sum_diff = 0;
        for (size_t i = 0; i < opencv_output.size(); ++i) {
            double diff = std::abs(opencv_output[i] - cuda_output[i]);
            max_diff = std::max(max_diff, diff);
            sum_diff += diff;
        }
        double avg_diff = sum_diff / opencv_output.size();
        
        std::cout << "  Max diff: " << std::fixed << std::setprecision(6) << max_diff << std::endl;
        std::cout << "  Avg diff: " << avg_diff << std::endl;
        std::cout << "  Status: " << (max_diff < 0.01 ? "PASS" : "CHECK") << std::endl;
    }
    
    return 0;
}
