//
// Created by root on 8/27/24.
//
#include "lite/lite.h"
#include "chrono"

static void test_default()
{
    auto start = std::chrono::high_resolution_clock::now();

    std::string engine_path = "/home/lite.ai.toolkit/examples/hub/onnx/sd/vae_model.onnx";

    lite::onnxruntime::sd::image_decoder::Vae *vae = new lite::onnxruntime::sd::image_decoder::Vae (engine_path);



    std::vector<std::string> input_vector = {"i am not good at cpp","goi ofg go !"};

    std::vector<std::vector<float>> output;

    vae->inference(input_vector,output);


    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "耗时: " << duration << " 毫秒" << std::endl;


    delete vae;

}



static void test_lite()
{
    test_default();
}

int main(__unused int argc, __unused char *argv[])
{
    test_lite();
    return 0;
}