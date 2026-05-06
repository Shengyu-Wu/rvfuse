// generic_ort_runner.cpp — Generic ONNX Runtime inference runner for perf profiling.
// Reads input shape from the model, generates random test data, runs inference.
// Usage: ./generic_ort_runner <model.onnx> [iterations]

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <cstdlib>

#include "onnxruntime_cxx_api.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> [iterations=10]" << std::endl;
        return 1;
    }

    const char* model_path = argv[1];
    int iterations = argc >= 3 ? std::atoi(argv[2]) : 10;
    if (iterations <= 0) iterations = 10;

    // Initialize ORT
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ort-perf");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    std::cout << "Loading model: " << model_path << std::endl;
    Ort::Session session(env, model_path, session_options);

    // Read input info from model
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
    std::string input_name = input_name_ptr.get();

    auto input_type_info = session.GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    auto input_shape = input_tensor_info.GetShape();
    ONNXTensorElementDataType input_type = input_tensor_info.GetElementType();

    // Print input info
    std::cout << "Input: " << input_name << " shape=[";
    size_t total_elements = 1;
    for (size_t i = 0; i < input_shape.size(); i++) {
        // Handle dynamic dimensions (negative values)
        if (input_shape[i] < 0) input_shape[i] = 1;
        std::cout << input_shape[i];
        if (i < input_shape.size() - 1) std::cout << ",";
        total_elements *= static_cast<size_t>(input_shape[i]);
    }
    std::cout << "] type=" << static_cast<int>(input_type) << std::endl;

    if (total_elements > 100'000'000) {
        std::cerr << "Warning: large input (" << total_elements << " elements)" << std::endl;
    }

    // Print output info
    auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    std::string output_name = output_name_ptr.get();
    std::cout << "Output: " << output_name << std::endl;

    // Generate random input data and create tensor matching the model's input type
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::mt19937 gen(42);
    Ort::Value input_tensor = [&]() -> Ort::Value {
        switch (input_type) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
                std::vector<float> data(total_elements);
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                for (auto& v : data) v = dist(gen);
                return Ort::Value::CreateTensor<float>(
                    memory_info, data.data(), total_elements,
                    input_shape.data(), input_shape.size());
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
                std::vector<double> data(total_elements);
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                for (auto& v : data) v = dist(gen);
                return Ort::Value::CreateTensor<double>(
                    memory_info, data.data(), total_elements,
                    input_shape.data(), input_shape.size());
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
                std::vector<int8_t> data(total_elements);
                std::uniform_int_distribution<int> dist(0, 127);
                for (auto& v : data) v = static_cast<int8_t>(dist(gen));
                return Ort::Value::CreateTensor<int8_t>(
                    memory_info, data.data(), total_elements,
                    input_shape.data(), input_shape.size());
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
                std::vector<uint8_t> data(total_elements);
                std::uniform_int_distribution<int> dist(0, 255);
                for (auto& v : data) v = static_cast<uint8_t>(dist(gen));
                return Ort::Value::CreateTensor<uint8_t>(
                    memory_info, data.data(), total_elements,
                    input_shape.data(), input_shape.size());
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
                std::vector<int32_t> data(total_elements);
                std::uniform_int_distribution<int32_t> dist(0, 100);
                for (auto& v : data) v = dist(gen);
                return Ort::Value::CreateTensor<int32_t>(
                    memory_info, data.data(), total_elements,
                    input_shape.data(), input_shape.size());
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
                std::vector<int64_t> data(total_elements);
                std::uniform_int_distribution<int64_t> dist(0, 100);
                for (auto& v : data) v = dist(gen);
                return Ort::Value::CreateTensor<int64_t>(
                    memory_info, data.data(), total_elements,
                    input_shape.data(), input_shape.size());
            }
            default:
                std::cerr << "Error: unsupported input type " << static_cast<int>(input_type)
                          << ". Supported: float, double, int8, uint8, int32, int64." << std::endl;
                std::exit(1);
        }
    }();

    // Warm-up run
    std::cout << "Warm-up run..." << std::endl;
    const char* input_names[] = {input_name.c_str()};
    const char* output_names[] = {output_name.c_str()};
    try {
        auto warmup_output = session.Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1);
        std::cout << "Warm-up OK" << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "Warm-up failed: " << e.what() << std::endl;
        return 1;
    }

    // Measured runs
    std::cout << "Running " << iterations << " iterations..." << std::endl;
    auto total_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        auto output = session.Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    std::cout << "Done: " << iterations << " iterations in " << total_ms << " ms" << std::endl;
    std::cout << "Average: " << total_ms / iterations << " ms/iter" << std::endl;

    return 0;
}
