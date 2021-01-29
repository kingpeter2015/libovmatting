
cmake_minimum_required(VERSION 3.10)
project(sample)

set(CMAKE_CXX_STANDARD 11)

set(TARGET_NAME "sample")

include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(ngraph REQUIRED OPTIONAL_COMPONENTS onnx_importer)
find_package(InferenceEngine REQUIRED)
find_package(OpenCV REQUIRED)

file(GLOB NATIVE_SRCS "${PROJECT_SOURCE_DIR}/src/*.cpp")
# set(SRC ${PROJECT_SOURCE_DIR}/src/*.cpp )

ADD_EXECUTABLE(${TARGET_NAME} ${NATIVE_SRCS})

target_compile_definitions(${TARGET_NAME} PRIVATE OPENCV_IMPORT_ENABLED)
target_link_libraries(${TARGET_NAME} PRIVATE opencv_core)

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_INFERENCE_EXTENSION_API)
target_link_libraries(${TARGET_NAME} PRIVATE IE::inference_engine ${NGRAPH_LIBRARIES})
target_link_libraries(${TARGET_NAME} PRIVATE ${ONNX_IMPORTER_LIBRARIES})
target_link_libraries(${TARGET_NAME} PRIVATE ${OpenCV_LIBS})
target_compile_definitions(${TARGET_NAME} PRIVATE NGRAPH_ONNX_IMPORT_ENABLED)