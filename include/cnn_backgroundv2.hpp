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

    std::string output_name_;
    int max_detections_count_ = 0;
    int object_size_ = 0;
    int enqueued_frames_ = 0;
    float width_ = 0;
    float height_ = 0;

public:
    explicit CNN_Background_V2(const CNNConfig& config);

    void submitRequest() override;
    void enqueue(const cv::Mat &frame, const cv::Mat& image) override;
    void wait() override { BaseAsyncCNN<MattingObject>::wait(); }
    void printPerformanceCounts(const std::string &fullDeviceName) override
    {
        BaseAsyncCNN<MattingObject>::printPerformanceCounts(fullDeviceName);
    }

    MattingObjects fetchResults() override;
};