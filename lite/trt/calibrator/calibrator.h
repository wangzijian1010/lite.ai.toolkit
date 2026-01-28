#include <NvInfer.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <iterator>

class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    // 构造函数
    Int8EntropyCalibrator(int batchSize, const std::string& calibrationDataPath,
                          const std::string& calibrationCacheFile,
                          int inputH, int inputW)
        : batchSize_(batchSize),
          calibrationCacheFile_(calibrationCacheFile),
          inputH_(inputH),
          inputW_(inputW) {

        // 1. 读取所有图片路径
        cv::glob(calibrationDataPath + "/*.jpg", filePaths_);

        // 【保险机制 1】数量熔断：强制只取前 100 张
        // 100 张对 YOLOv5 的校准精度已经足够了，跑太多纯属浪费时间
        int max_calib_images = 100;
        if (filePaths_.size() > max_calib_images) {
            std::cout << "[Info] Too many images (" << filePaths_.size() << "), keeping first " << max_calib_images << " for fast calibration." << std::endl;
            filePaths_.resize(max_calib_images);
        }

        if (filePaths_.empty()) {
            std::cerr << "[Warning] No images found in " << calibrationDataPath << std::endl;
        } else {
            std::cout << "[Info] Calibration prepared with " << filePaths_.size() << " images." << std::endl;
        }

        inputCount_ = batchSize * 3 * inputH_ * inputW_;
        inputSize_ = inputCount_ * sizeof(float);
        cudaMalloc(&deviceInput_, inputSize_);
    }

    ~Int8EntropyCalibrator() {
        if (deviceInput_) cudaFree(deviceInput_);
    }

    int getBatchSize() const noexcept override {
        return batchSize_;
    }

    // TensorRT 会反复调用这个函数
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override {
        // 如果图片读完了，返回 false，TensorRT 就会停止校准
        if (currentImgIdx_ >= filePaths_.size()) return false;

        std::vector<float> batchData(inputCount_);

        // 【保险机制 2】打印心跳：显示当前进度，并强制刷新(std::flush)确保能看到
        std::cout << "[Calibration] Processing image " << currentImgIdx_
                  << " / " << filePaths_.size() << " ... " << std::flush;

        for (int i = 0; i < batchSize_ && currentImgIdx_ < filePaths_.size(); ++i) {
            // 读取图片
            cv::Mat img = cv::imread(filePaths_[currentImgIdx_]);

            // 【保险机制 3】死循环克星：必须在这里自增索引！
            // 很多“假死”都是因为忘了这句，导致永远在读第 0 张图
            currentImgIdx_++;

            if (img.empty()) {
                std::cout << "Read failed!" << std::endl;
                continue;
            }

            // --- 预处理逻辑 (Letterbox + Normalize + CHW) ---
            int img_height = img.rows;
            int img_width = img.cols;
            float w_r = (float)inputW_ / (float)img_width;
            float h_r = (float)inputH_ / (float)img_height;
            float r = std::min(w_r, h_r);

            int new_unpad_w = static_cast<int>((float)img_width * r);
            int new_unpad_h = static_cast<int>((float)img_height * r);
            int dw = (inputW_ - new_unpad_w) / 2;
            int dh = (inputH_ - new_unpad_h) / 2;

            cv::Mat mat_rs(inputH_, inputW_, CV_8UC3, cv::Scalar(114, 114, 114));
            cv::Mat new_unpad_mat;
            cv::resize(img, new_unpad_mat, cv::Size(new_unpad_w, new_unpad_h));
            new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));

            cv::Mat canvas;
            cv::cvtColor(mat_rs, canvas, cv::COLOR_BGR2RGB);
            canvas.convertTo(canvas, CV_32F, 1.0 / 255.0);

            int volImg = inputH_ * inputW_;
            int offset = i * 3 * volImg;

            for (int h = 0; h < inputH_; ++h) {
                for (int w = 0; w < inputW_; ++w) {
                    cv::Vec3f pixel = canvas.at<cv::Vec3f>(h, w);
                    batchData[offset + 0 * volImg + h * inputW_ + w] = pixel[0];
                    batchData[offset + 1 * volImg + h * inputW_ + w] = pixel[1];
                    batchData[offset + 2 * volImg + h * inputW_ + w] = pixel[2];
                }
            }
        }

        cudaMemcpy(deviceInput_, batchData.data(), inputSize_, cudaMemcpyHostToDevice);
        bindings[0] = deviceInput_;

        std::cout << "Done." << std::endl; // 打印完成标记
        return true;
    }

    const void* readCalibrationCache(size_t& length) noexcept override {
        calibrationCache_.clear();
        std::ifstream input(calibrationCacheFile_, std::ios::binary);
        input >> std::noskipws;
        if (input.good()) {
            std::copy(std::istream_iterator<char>(input),
                      std::istream_iterator<char>(),
                      std::back_inserter(calibrationCache_));
        }
        length = calibrationCache_.size();
        return length ? calibrationCache_.data() : nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length) noexcept override {
        std::ofstream output(calibrationCacheFile_, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
        std::cout << "[Info] Calibration table written to: " << calibrationCacheFile_ << std::endl;
    }

private:
    int batchSize_;
    int inputH_;
    int inputW_;
    int inputCount_;
    size_t inputSize_;
    int currentImgIdx_ = 0; // 这个变量控制进度

    std::string calibrationCacheFile_;
    std::vector<std::string> filePaths_;
    void* deviceInput_ = nullptr;
    std::vector<char> calibrationCache_;
};