// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cnn_backgroundv2.hpp"

#include <algorithm>
#include <string>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <inference_engine.hpp>

#include "ovmatter.h"

using namespace InferenceEngine;

CNN_Background_V2::CNN_Background_V2(const CNNConfig& config) : _config(config), _bBgr(false)
{
    topoName = "CNN_Background_V2";
    isAsync = config.is_async;

    cnn_network_ = _config.ie.ReadNetwork(_config.path_to_model, _config.path_to_bin);

    load();
}

bool CNN_Background_V2::isBgrEnqueued()
{
    return _bBgr;
}

void CNN_Background_V2::load()
{
    InferenceEngine::InputsDataMap inputInfo = cnn_network_.getInputsInfo();
    const int currentBatchSize = cnn_network_.getBatchSize();
    if (currentBatchSize != _config.max_batch_size)
    {
        cnn_network_.setBatchSize(_config.max_batch_size);
    }

    int nInputInfoSize = inputInfo.size();
    auto inInfo = cnn_network_.getInputsInfo();
    std::map<std::string, InferenceEngine::SizeVector> input_shapes;
    for (auto& item : inInfo)
    {
        auto input_data = item.second;
        input_data->setPrecision(Precision::FP32);
        input_data->setLayout(Layout::NCHW);
        input_data->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
        input_data->getPreProcess().setColorFormat(ColorFormat::RGB);

        SizeVector input_dims = input_data->getInputData()->getTensorDesc().getDims();
        input_dims[3] = _config.input_shape.width;
        input_dims[2] = _config.input_shape.height;
        input_shapes[item.first] = input_dims;
    }
    cnn_network_.reshape(input_shapes);

    InferenceEngine::OutputsDataMap outputInfo = cnn_network_.getOutputsInfo();
    for (auto& item : outputInfo)
    {
        item.second->setPrecision(Precision::FP32);
        Layout layout = item.second->getLayout();
        if (layout != Layout::NCHW)
        {
            item.second->setLayout(Layout::NC);
        }
        else
        {
            item.second->setLayout(Layout::NCHW);
        }
    }

    if (_config.cpu_threads_num > 0)
    {
        std::map<std::string, std::string> loadParams;
        loadParams[PluginConfigParams::KEY_CPU_THREADS_NUM] = std::to_string(_config.cpu_threads_num);
        loadParams[PluginConfigParams::KEY_CPU_BIND_THREAD] = _config.cpu_bind_thread ? PluginConfigParams::YES : PluginConfigParams::NO;
        loadParams[PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS] = std::to_string(_config.cpu_throughput_streams);
        net_ = _config.ie.LoadNetwork(cnn_network_, _config.deviceName, loadParams);
    }
    else
    {
        net_ = _config.ie.LoadNetwork(cnn_network_, _config.deviceName);
    }

    request = net_.CreateInferRequestPtr();

    _bBgr = false;
}

void CNN_Background_V2::reshape(cv::Size input_shape)
{
    _config.input_shape = input_shape;
    load();
}

void CNN_Background_V2::submitRequest()
{
    if (!request)
    {
        return;
    }

    BaseAsyncCNN<MattingObject>::submitRequest();
}

void CNN_Background_V2::enqueue(const std::string& name, const cv::Mat& frame)
{
    if (!request)
    {
        return;
    }

    cv::Mat matFrame = frame.clone();
    if (matFrame.rows != _config.input_shape.height || matFrame.cols != _config.input_shape.width)
    {
        cv::resize(matFrame, matFrame, _config.input_shape);
    }
    unsigned char* dataMatFrame = matFrame.data;

    Blob::Ptr blob = request->GetBlob(name);
    auto data = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
    if (data == nullptr)
    {
        std::string errorName = "Input blob (" + name;
        errorName += ") has not allocated buffer";
        throw std::runtime_error(errorName);
        //throw std::runtime_error("Input blob has not allocated buffer");
    }
    size_t num_channels = blob->getTensorDesc().getDims()[1];
    size_t image_width = blob->getTensorDesc().getDims()[3];
    size_t image_height = blob->getTensorDesc().getDims()[2];
    size_t image_size = image_width * image_height;

    unsigned char* imagesData = matFrame.data;

    /** Iterate over all pixel in image (b,g,r) **/
    for (size_t pid = 0; pid < image_size; pid++)
    {
        /** Iterate over all channels **/
        int numC = pid * num_channels;
        *(data + pid) = *(imagesData + numC + 2) / 255.0f;
        *(data + image_size + pid) = *(imagesData + numC + 1) / 255.0f;
        *(data + image_size + image_size + pid) = *(imagesData + numC) / 255.0f;
    }

    if (name == "bgr")
    {
        _bBgr = true;
    }
}

void CNN_Background_V2::enqueueAll(const cv::Mat& frame, const cv::Mat& bgr)
{
    try
    {
        //1.init fields
        enqueue("src", frame);
        enqueue("bgr", bgr);
    }
    catch (const std::exception& e)
    {
        std::cerr << "CNN_Background_V2::enqueue():" << e.what() << '\n';
    }    
}

MattingObjects CNN_Background_V2::fetchResults()
{
    MattingObjects results;

    if (!request)
    {
        return results;
    }

    try
    {
        cv::Mat matPha = cv::Mat::zeros(_config.input_shape, CV_8UC1);
        unsigned char* dataMatPha = matPha.data;

        //3.1. get alpha mat
        Blob::Ptr blobPha = request->GetBlob("pha");
        auto dataPha = blobPha->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();

        size_t num_channels_pha = blobPha->getTensorDesc().getDims()[1];
        size_t image_width = blobPha->getTensorDesc().getDims()[3];
        size_t image_height = blobPha->getTensorDesc().getDims()[2];
        unsigned long image_size = image_width * image_height;        
        for (size_t pid = 0; pid < image_size; pid++)
        {
            float alpha = *(dataPha + pid);
            *(dataMatPha + pid) = alpha * 255.0f;
        }

        MattingObject objMatter;
        objMatter.pha = matPha.clone();
        results.push_back(objMatter);

        return results;
    }
    catch (const std::exception& e)
    {
        std::cerr << "CNN_Background_V2::fetchResults():" << e.what() << '\n';
        return results;
    }
}

