cmake_minimum_required(VERSION 3.16)
project(yolo_runner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime installation")

add_executable(yolo_inference yolo_runner.cpp)

target_include_directories(yolo_inference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ONNXRUNTIME_ROOT}/include
)

target_link_libraries(yolo_inference PRIVATE
    ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.a
    Threads::Threads
    m dl
)

set_target_properties(yolo_inference PROPERTIES LINK_FLAGS "-static")
