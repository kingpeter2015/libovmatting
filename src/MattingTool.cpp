// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "MattingTool.hpp"

#include <algorithm>
#include <string>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <inference_engine.hpp>

#include <ngraph/ngraph.hpp>

using namespace InferenceEngine;

void MattingTool::submitRequest()
{
    if (!enqueued_frames_)
        return;
    enqueued_frames_ = 0;
    BaseMatting::submitRequest();
}

void MattingTool::enqueue(const cv::Mat &frame, const cv::Mat &image)
{
    if (!request)
    {
        request = net_.CreateInferRequestPtr();
    }

    width_ = static_cast<float>(frame.cols);
    height_ = static_cast<float>(frame.rows);

    Blob::Ptr inputFrame = request->GetBlob("src");
    matU8ToBlob<uint8_t>(frame, inputFrame);

    Blob::Ptr inputImage = request->GetBlob("bgr");
    matU8ToBlob<uint8_t>(image, inputImage);

    enqueued_frames_ = 1;
}

MattingTool::MattingTool(const CnnConfig &config) : BaseMatting(config.is_async), _config(config)
{
    topoName = "MattingTool";
    auto cnnNetwork = _config.ie.ReadNetwork(config.path_to_model, config.path_to_bin);

    InferenceEngine::InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    const int currentBatchSize = cnnNetwork.getBatchSize();
    if (currentBatchSize != _config.max_batch_size)
    {
        cnnNetwork.setBatchSize(_config.max_batch_size);
    }

    int nInputInfoSize = inputInfo.size();
    auto inInfo = cnnNetwork.getInputsInfo();
    std::map<std::string, InferenceEngine::SizeVector> input_shapes;
    for (auto &item : inInfo)
    {
        auto input_data = item.second;
        input_data->setPrecision(Precision::U8);
        input_data->setLayout(Layout::NCHW);
        input_data->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
        input_data->getPreProcess().setColorFormat(ColorFormat::BGR);

        SizeVector input_dims = input_data->getInputData()->getTensorDesc().getDims();
        input_shapes[item.first] = input_dims;
    }
    cnnNetwork.reshape(input_shapes);

    InferenceEngine::OutputsDataMap outputInfo(cnnNetwork.getOutputsInfo());
    for (auto &item : outputInfo)
    {
        item.second->setPrecision(Precision::FP32);
        if (item.first == "fgr")
        {
            item.second->setLayout(Layout::NCHW);
        }
        else
        {
            item.second->setLayout(TensorDesc::getLayoutByDims(item.second->getDims()));
        }
    }

    if (config.networkCfg.nCpuThreadsNum > 0)
    {
        std::map<std::string, std::string> loadParams;
        loadParams[PluginConfigParams::KEY_CPU_THREADS_NUM] = std::to_string(config.networkCfg.nCpuThreadsNum);
        loadParams[PluginConfigParams::KEY_CPU_BIND_THREAD] = config.networkCfg.bCpuBindThread ? PluginConfigParams::YES : PluginConfigParams::NO;
        loadParams[PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS] = std::to_string(config.networkCfg.nCpuThroughputStreams);
        net_ = _config.ie.LoadNetwork(cnnNetwork, _config.deviceName, loadParams);
    }
    else
    {
        net_ = _config.ie.LoadNetwork(cnnNetwork, _config.deviceName);
    }
}

MattingObjects MattingTool::fetchResults()
{
    MattingObjects results;

    InferenceEngine::Blob::Ptr blobFgr = request->GetBlob("fgr");
    if (blobFgr == nullptr)
    {
        THROW_IE_EXCEPTION << "MattingTool::fetchResults() Invalid fgr blob '";
    }

    return results;
}
