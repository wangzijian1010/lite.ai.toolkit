# calibrator/CMakeLists.txt

# ============================================================
# 1. 原有的 lite_builder (保留兼容)
# ============================================================
add_executable(lite_builder build.cpp)
target_include_directories(lite_builder PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lite_builder
        PRIVATE
        ${OpenCV_LIBS}
        nvinfer
        nvonnxparser
        cudart
)

# ============================================================
# 2. 新增: build_engines - 支持 FP32/FP16/INT8 一键构建
# ============================================================
add_executable(build_engines build_engines.cpp)
target_include_directories(build_engines PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(build_engines
        PRIVATE
        ${OpenCV_LIBS}
        nvinfer
        nvonnxparser
        cudart
)

# ============================================================
# 安装
# ============================================================
install(TARGETS lite_builder build_engines DESTINATION bin)