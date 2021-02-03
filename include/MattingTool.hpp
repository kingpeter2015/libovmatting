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

class MattingTool : public AsyncDetection<MattingObject>, public BaseMatting
{
private:
    CnnConfig _config;
    InferenceEngine::ExecutableNetwork net_;

    std::string output_name_;
    int max_detections_count_ = 0;
    int object_size_ = 0;
    int enqueued_frames_ = 0;
    float width_ = 0;
    float height_ = 0;

public:
    explicit MattingTool(const CnnConfig &config);

    void submitRequest() override;
    void enqueue(const cv::Mat &frame, const cv::Mat& image) override;
    void wait() override { BaseMatting::wait(); }
    void printPerformanceCounts(const std::string &fullDeviceName) override
    {
        BaseMatting::printPerformanceCounts(fullDeviceName);
    }

    MattingObjects fetchResults() override;
};