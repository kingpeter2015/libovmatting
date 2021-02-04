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

#include <ngraph/ngraph.hpp>

using namespace InferenceEngine;

CNN_Background_V2::CNN_Background_V2(const CNNConfig& config) : _config(config), _bEnqueue(false)
{
    topoName = "CNN_Background_V2";
    isAsync = config.is_async;

    auto cnnNetwork = _config.ie.ReadNetwork(config.path_to_model, config.path_to_bin);

    InferenceEngine::InputsDataMap inputInfo = cnnNetwork.getInputsInfo();
    const int currentBatchSize = cnnNetwork.getBatchSize();
    if (currentBatchSize != _config.max_batch_size)
    {
        cnnNetwork.setBatchSize(_config.max_batch_size);
    }

    int nInputInfoSize = inputInfo.size();
    auto inInfo = cnnNetwork.getInputsInfo();
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
    cnnNetwork.reshape(input_shapes);

    InferenceEngine::OutputsDataMap outputInfo = cnnNetwork.getOutputsInfo();
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
        net_ = _config.ie.LoadNetwork(cnnNetwork, _config.deviceName, loadParams);
    }
    else
    {
        net_ = _config.ie.LoadNetwork(cnnNetwork, _config.deviceName);
    }
}

void CNN_Background_V2::submitRequest()
{
    if (!_bEnqueue)
    {
        return;
    }
    _frame_count++;
    BaseAsyncCNN<MattingObject>::submitRequest();
}

void CNN_Background_V2::enqueue(const cv::Mat& frame, const cv::Mat& bgr, const cv::Mat& bgrReplace, const cv::Size& out_shape)
{
    if (!request)
    {
        request = net_.CreateInferRequestPtr();
    }

    try
    {
        //1.init fields
        _frame = frame.clone();
        _bgr = bgr.clone();
        _bgrReplace = bgrReplace.clone();
        _out_shape = out_shape;

        //2. Input
        cv::Mat matFrame = _frame.clone();
        cv::Mat matBgr = _bgr.clone();
        if (matFrame.rows != _config.input_shape.height || matFrame.cols != _config.input_shape.width)
        {
            cv::resize(matFrame, matFrame, _config.input_shape);
        }

        if (matBgr.rows != _config.input_shape.height || matBgr.cols != _config.input_shape.width)
        {
            cv::resize(matBgr, matBgr, _config.input_shape);
        }

        std::vector<std::string> vecBlockNames;
        vecBlockNames.emplace_back("src");
        vecBlockNames.emplace_back("bgr");

        unsigned char* dataSrc = (matFrame.data);
        unsigned char* dataBgr = (matBgr.data);
        int count = -1;
        for (auto& blockName : vecBlockNames)
        {
            count++;

            Blob::Ptr blob = request->GetBlob(blockName);
            auto data = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
            if (data == nullptr)
            {
                throw std::runtime_error("Input blob has not allocated buffer");
            }
            size_t num_channels = blob->getTensorDesc().getDims()[1];
            size_t image_width = blob->getTensorDesc().getDims()[3];
            size_t image_height = blob->getTensorDesc().getDims()[2];
            size_t image_size = image_width * image_height;

            /** Iterate over all input images **/
            unsigned char* imagesData = count == 0 ? dataSrc : dataBgr;

            /** Iterate over all pixel in image (b,g,r) **/
            for (size_t pid = 0; pid < image_size; pid++)
            {
                /** Iterate over all channels **/
                int numC = pid * num_channels;
                *(data + pid) = *(imagesData + numC + 2) / 255.0f;
                *(data + image_size + pid) = *(imagesData + numC + 1) / 255.0f;
                *(data + image_size + image_size + pid) = *(imagesData + numC) / 255.0f;
            }
        }

        _bEnqueue = true;
    }
    catch (const std::exception& e)
    {
        _bEnqueue = false;
        std::cerr << "CNN_Background_V2::enqueue():" << e.what() << '\n';
    }
    
}



MattingObjects CNN_Background_V2::fetchResults()
{
    MattingObjects results;

    try
    {
        cv::Mat matPha = cv::Mat::zeros(_config.input_shape, CV_8UC1);
        cv::Mat matCom = cv::Mat::zeros(_out_shape, CV_8UC3);
        unsigned char* dataMatPha = matPha.data;
        unsigned char* dataMatCom = matCom.data;

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

        if (matPha.rows != _out_shape.height || matPha.cols != _out_shape.width)
        {
            cv::resize(matPha, matPha, _out_shape, 0, 0, cv::INTER_CUBIC);
        }
        dataMatPha = matPha.data;
        image_size = matPha.rows * matPha.cols;

        //3.2 get bgr2
        cv::Mat matBgr2 = _bgrReplace.clone();
        if (matBgr2.rows != _out_shape.height || matBgr2.cols != _out_shape.width)
        {
            cv::resize(matBgr2, matBgr2, _out_shape);
        }
        unsigned char* dataMatBgr2 = matBgr2.data;

        //3.3 get frame
        cv::Mat matFrame1 = _frame.clone();
        if (matFrame1.rows != _out_shape.height || matFrame1.cols != _out_shape.width)
        {
            cv::resize(matFrame1, matFrame1, _out_shape);
        }
        unsigned char* dataMatFrame1 = matFrame1.data;

        //3.4 get composite
        int num_channels = matCom.channels();
        for (size_t pid = 0; pid < image_size; pid++)
        {
            int nAlpha = *(dataMatPha + pid);
            int rowC = pid * num_channels;
            if (nAlpha == 0)
            {
                *(dataMatCom + rowC + 2) = *(dataMatBgr2 + rowC + 2);
                *(dataMatCom + rowC + 1) = *(dataMatBgr2 + rowC + 1);
                *(dataMatCom + rowC) = *(dataMatBgr2 + rowC);
            }
            else if (nAlpha == 255)
            {
                *(dataMatCom + rowC + 2) = *(dataMatFrame1 + rowC + 2);
                *(dataMatCom + rowC + 1) = *(dataMatFrame1 + rowC + 1);
                *(dataMatCom + rowC) = *(dataMatFrame1 + rowC);
            }
            else
            {
                float alpha = nAlpha / 255.0;
                *(dataMatCom + rowC + 2) = *(dataMatFrame1 + rowC + 2) * alpha + *(dataMatBgr2 + rowC + 2) * (1 - alpha);
                *(dataMatCom + rowC + 1) = *(dataMatFrame1 + rowC + 1) * alpha + *(dataMatBgr2 + rowC + 1) * (1 - alpha);
                *(dataMatCom + rowC) = *(dataMatFrame1 + rowC) * alpha + *(dataMatBgr2 + rowC) * (1 - alpha);
            }
        }

        MattingObject objMatter;
        objMatter.com = matCom.clone();
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
