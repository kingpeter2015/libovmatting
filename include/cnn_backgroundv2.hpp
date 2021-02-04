// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

#include "cnn.hpp"


struct MattingObject {
    cv::Mat pha;
    cv::Mat fgr;
    cv::Mat com;
};

using MattingObjects = std::vector<MattingObject>;

class CNN_Background_V2 : public BaseAsyncCNN<MattingObject>
{
private:
    CNNConfig _config;
    InferenceEngine::ExecutableNetwork net_;
    cv::Mat _frame, _bgr, _bgrReplace;
    cv::Size _out_shape;

    int _frame_count = 0;
    bool _bEnqueue;

public:
    explicit CNN_Background_V2(const CNNConfig& config);

    void submitRequest() override;
    void enqueue(const cv::Mat& frame, const cv::Mat& bgr, const cv::Mat& bgrReplace, const cv::Size& out_shape) override;
    void wait() override { BaseAsyncCNN<MattingObject>::wait(); }
    void printPerformanceCounts(const std::string &fullDeviceName) override
    {
        BaseAsyncCNN<MattingObject>::printPerformanceCounts(fullDeviceName);
    }

    MattingObjects fetchResults() override;
    CNNConfig* getConfig()
    {
        return &_config;
    }
};


class CNN_NUllMatting : public BaseAsyncCNN<MattingObject> {
public:
    explicit CNN_NUllMatting() {}
    void enqueue(const cv::Mat& frame, const cv::Mat& bgr, const cv::Mat& bgrReplace, const cv::Size& out_shape) override {}
    void submitRequest() override {}
    void wait() override {}
    void printPerformanceCounts(const std::string&) override {}
    MattingObjects fetchResults() override { return {}; }
};