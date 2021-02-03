
cmake_minimum_required(VERSION 3.10)
project(libovmatting)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) #-fPIC
set(LIB_OVMATTER_EXPORTS 1) 

#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/../out/debug)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/../out/release)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../build)

set(TARGET_NAME "ovmatting")

include_directories(${PROJECT_SOURCE_DIR}/../include)
message("Root Directory:${PROJECT_SOURCE_DIR}")

find_package(ngraph REQUIRED OPTIONAL_COMPONENTS onnx_importer)
find_package(InferenceEngine REQUIRED)
find_package(OpenCV REQUIRED)

file(GLOB NATIVE_SRCS "${PROJECT_SOURCE_DIR}/../src/*.cpp")
message("Source Code Files:${NATIVE_SRCS}")
# set(SRC ${PROJECT_SOURCE_DIR}/src/*.cpp )

ADD_LIBRARY (${TARGET_NAME} SHARED ${NATIVE_SRCS})

target_compile_definitions(${TARGET_NAME} PRIVATE OPENCV_IMPORT_ENABLED)
target_link_libraries(${TARGET_NAME} PRIVATE opencv_core)

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_INFERENCE_EXTENSION_API)
target_link_libraries(${TARGET_NAME} PRIVATE IE::inference_engine ${NGRAPH_LIBRARIES})
target_link_libraries(${TARGET_NAME} PRIVATE ${ONNX_IMPORTER_LIBRARIES})
target_link_libraries(${TARGET_NAME} PRIVATE ${OpenCV_LIBS})
target_compile_definitions(${TARGET_NAME} PRIVATE NGRAPH_ONNX_IMPORT_ENABLED)