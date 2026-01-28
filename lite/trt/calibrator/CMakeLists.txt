# calibrator/CMakeLists.txt

# 1. 定义可执行文件，源文件是 build.cpp
# 注意：calibrator.h 是头文件，不需要加到 add_executable 中，只要 build.cpp include 了它就行
add_executable(lite_builder build.cpp)

# 2. 包含当前目录的头文件 (这样 build.cpp 才能找到 calibrator.h)
target_include_directories(lite_builder PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# 3. 链接依赖库
# 注意：OpenCV_LIBS 和 CUDA_LIBRARIES 这些变量会从主项目继承过来
target_link_libraries(lite_builder
        PRIVATE
        ${OpenCV_LIBS}       # OpenCV 库
        nvinfer              # TensorRT 核心库
        nvonnxparser         # ONNX 解析库
        cudart               # CUDA Runtime
        # 如果你的主项目定义了其他变量，也可以加在这里
)

# 4. (可选) 安装路径：编译完后把 lite_builder 放到 install/bin 目录下，方便你运行
install(TARGETS lite_builder DESTINATION bin)