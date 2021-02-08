// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

#include "cnn.hpp"


class CNN_Background_V2 : public BaseAsyncCNN<MattingObject>
{
private:
    CNNConfig _config;
    InferenceEngine::ExecutableNetwork net_;
    InferenceEngine::CNNNetwork cnn_network_;

    bool _bBgr;

private:
    void load();

public:
    explicit CNN_Background_V2(const CNNConfig& config);
    void reshape(cv::Size input_shape);

    void submitRequest() override;
    void enqueueAll(const cv::Mat& frame, const cv::Mat& bgr);
    void enqueue(const std::string& name, const cv::Mat& frame) override;
    void wait() override { BaseAsyncCNN<MattingObject>::wait(); }
    void printPerformanceCounts(const std::string &fullDeviceName) override
    {
        BaseAsyncCNN<MattingObject>::printPerformanceCounts(fullDeviceName);
    }

    MattingObjects fetchResults() override;
    bool isBgrEnqueued();
};
