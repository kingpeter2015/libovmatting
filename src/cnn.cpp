// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cnn.hpp"

#include <string>
#include <vector>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <inference_engine.hpp>
#include "ns_utils.hpp"

/*
const float meanValues[] = {static_cast<const float>(FLAGS_mean_val_r),
                                    static_cast<const float>(FLAGS_mean_val_g),
                                    static_cast<const float>(FLAGS_mean_val_b)};
                                    */

FaceTimerCounter::FaceTimerCounter() {}

FaceTimerCounter::~FaceTimerCounter()
{
}

void FaceTimerCounter::Start()
{
    _elapse = 0;
    _start = std::chrono::high_resolution_clock::now();
    _started = true;
}

int64_t FaceTimerCounter::Elapse()
{
    auto elapsed = std::chrono::high_resolution_clock::now() - _start;
    _elapse = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

    return _elapse;
}

using namespace InferenceEngine;

CnnDLSDKBase::CnnDLSDKBase(const Config &config) : _config(config) {}

void CnnDLSDKBase::Load()
{
    auto cnnNetwork = _config.ie.ReadNetwork(_config.path_to_model, _config.path_to_bin);

    auto input_shapes = cnnNetwork.getInputShapes();

    const int currentBatchSize = cnnNetwork.getBatchSize();
    if (currentBatchSize != _config.max_batch_size)
        cnnNetwork.setBatchSize(_config.max_batch_size);

    _inInfo = cnnNetwork.getInputsInfo();
    _input_blob_names.clear();
    _input_shapes.clear();
    for (auto &item : _inInfo)
    {
        auto input_data = item.second;
        _input_blob_names.push_back(item.first);
        input_data->setPrecision(Precision::FP32);
        Layout layout = input_data->getLayout();
        std::cout << "Input Name:" << item.first << ", Layout Type:" << layout << std::endl;

        input_data->setLayout(Layout::NCHW);
        input_data->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
        input_data->getPreProcess().setColorFormat(ColorFormat::RGB);

        SizeVector input_dims = input_data->getInputData()->getTensorDesc().getDims();
        _input_shapes[item.first] = input_dims;
    }

    cnnNetwork.reshape(_input_shapes);

    _outInfo = cnnNetwork.getOutputsInfo();
    _output_blobs_names.clear();
    for (auto &item : _outInfo)
    {
        item.second->setPrecision(Precision::FP32);
        _output_blobs_names.push_back(item.first);

        Layout layout = item.second->getLayout();
        std::cout << "Output Name:" << item.first << ", Layout Type:" << layout << std::endl;

        item.second->setLayout(Layout::NCHW);
    }

    //std::map<std::string, std::string> config;
    //config[PluginConfigParams::KEY_CPU_THREADS_NUM] = "4";
    //config[PluginConfigParams::KEY_CPU_BIND_THREAD] =  PluginConfigParams::YES;
    //config[PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS] = "4";
    _executable_network_ = _config.ie.LoadNetwork(cnnNetwork, _config.deviceName);
    /*

    if (_config.networkCfg.nCpuThreadsNum > 0)
    {
        std::map<std::string, std::string> loadParams;
        loadParams[PluginConfigParams::KEY_CPU_THREADS_NUM] = std::to_string(_config.networkCfg.nCpuThreadsNum);
        loadParams[PluginConfigParams::KEY_CPU_BIND_THREAD] = _config.networkCfg.bCpuBindThread ? PluginConfigParams::YES : PluginConfigParams::NO;
        loadParams[PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS] = std::to_string(_config.networkCfg.nCpuThroughputStreams);
        _executable_network_ = _config.ie.LoadNetwork(cnnNetwork, _config.deviceName, loadParams);
    }
    else
    {
        
    }
    */

    _infer_request_ = _executable_network_.CreateInferRequest();
}

MattingCNN::MattingCNN(const Config &config)
    : CnnDLSDKBase(config)
{
    Load();
}

void MattingCNN::Compute(const cv::Mat &frame, cv::Mat &bgr, std::map<std::string, cv::Mat> *result, cv::Size &outp_shape) const
{
    try
    {
        //1.Input
        cv::Mat iBgr, iFrame,matBgr, matFrame;
        iBgr = bgr.clone();
        iFrame = frame.clone();

        if(frame.rows != outp_shape.height || frame.cols != outp_shape.width)
        {
            cv::resize(frame, iFrame, outp_shape);
        }
        
        if(bgr.rows != outp_shape.height || bgr.cols != outp_shape.width)
        {
            cv::resize(bgr, iBgr, outp_shape);
        }

        matBgr = iBgr.clone();
        matFrame = iFrame.clone();
        std::vector<std::string> vecBlockNames;
        vecBlockNames.emplace_back("src");
        vecBlockNames.emplace_back("bgr");

        unsigned char *dataSrc = (matFrame.data);
        unsigned char *dataBgr = (matBgr.data);

        //Benchmarking Input
        {
            TimerCounter tCount("Phase1-Inputting");
            int count = -1;
            for (auto &blockName : vecBlockNames)
            {
                count++;

                Blob::Ptr blob = _infer_request_.GetBlob(blockName);
                auto data = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
                if (data == nullptr)
                {
                    throw std::runtime_error("Input blob has not allocated buffer");
                }
                size_t num_channels = blob->getTensorDesc().getDims()[1];
                size_t image_size = blob->getTensorDesc().getDims()[3] * blob->getTensorDesc().getDims()[2];
                /** Iterate over all input images **/
                unsigned char *imagesData = count == 0 ? dataSrc : dataBgr;
                /** Iterate over all pixel in image (b,g,r) **/
                for (size_t pid = 0; pid < image_size; pid++)
                {
                    /** Iterate over all channels **/
                    data[0 * image_size + pid] = imagesData[pid * num_channels + 2] / 255.0;
                    data[1 * image_size + pid] = imagesData[pid * num_channels + 1] / 255.0;
                    data[2 * image_size + pid] = imagesData[pid * num_channels + 0] / 255.0;
                }
            }
        }

        //2.Infer
        {
            TimerCounter tCount("Phase2-Infer()");
            _infer_request_.Infer();
        }

        //3.Output
        //outPha
        {
            TimerCounter tCount("Phase3-Outputting");
            Blob::Ptr outPha = _infer_request_.GetBlob("pha");
            InferenceEngine::SizeVector dimsPha = outPha->getTensorDesc().getDims();
            std::vector<int> phaSizes(dimsPha.size(), 0);
            for (size_t i = 0; i < phaSizes.size(); ++i)
            {
                phaSizes[i] = dimsPha[i];
            }
            cv::Mat matPha(phaSizes, CV_32F, outPha->buffer());
            matPha = matPha.reshape(phaSizes[1], {outp_shape.height, outp_shape.width});
            matPha.convertTo(matPha, CV_8U, 255.0);

            cv::Mat bgr_green(outp_shape.height, outp_shape.width, CV_8UC3, cv::Scalar(120, 255, 155));
            cv::Mat pha_sup = bgr_green.clone();
            cv::Mat com_fgr = frame.clone();
            cv::Mat com = cv::Mat::zeros(com_fgr.size(), com_fgr.type());
            cv::Mat matFgr = cv::Mat::zeros(com_fgr.size(), com_fgr.type());

            int channels = matFgr.channels();
            int image_size = matFgr.rows * matFgr.cols;
            for (int i = 0; i < image_size; i++)
            {
                float alpha = matPha.data[i + 0] / 255.0;
                com.data[i * channels + 0] = com_fgr.data[i * channels + 0] * alpha + pha_sup.data[i * channels + 0] * (1.0 - alpha);
                com.data[i * channels + 1] = com_fgr.data[i * channels + 1] * alpha + pha_sup.data[i * channels + 1] * (1.0 - alpha);
                com.data[i * channels + 2] = com_fgr.data[i * channels + 2] * alpha + pha_sup.data[i * channels + 2] * (1.0 - alpha);

                matFgr.data[i * channels + 0] = com_fgr.data[i * channels + 0] * alpha;
                matFgr.data[i * channels + 1] = com_fgr.data[i * channels + 1] * alpha;
                matFgr.data[i * channels + 2] = com_fgr.data[i * channels + 2] * alpha;
            }
            /*
            for (int i = 0; i < matPha.rows; i++)
            {
                for (int j = 0; j < matPha.cols; j++)
                {
                    float alpha = matPha.at<uint8_t>(i, j) / 255.0;
                    com.at<cv::Vec3b>(i, j)[0] = com_fgr.at<cv::Vec3b>(i, j)[0] * alpha + pha_sup.at<cv::Vec3b>(i, j)[0] * (1.0 - alpha);
                    com.at<cv::Vec3b>(i, j)[1] = com_fgr.at<cv::Vec3b>(i, j)[1] * alpha + pha_sup.at<cv::Vec3b>(i, j)[1] * (1.0 - alpha);
                    com.at<cv::Vec3b>(i, j)[2] = com_fgr.at<cv::Vec3b>(i, j)[2] * alpha + pha_sup.at<cv::Vec3b>(i, j)[2] * (1.0 - alpha);

                    matFgr.at<cv::Vec3b>(i, j)[0] = com_fgr.at<cv::Vec3b>(i, j)[0] * alpha;
                    matFgr.at<cv::Vec3b>(i, j)[1] = com_fgr.at<cv::Vec3b>(i, j)[1] * alpha;
                    matFgr.at<cv::Vec3b>(i, j)[2] = com_fgr.at<cv::Vec3b>(i, j)[2] * alpha;
                }
            }
            */

            (*result)["com"] = com;
            (*result)["fgr"] = matFgr;
            (*result)["pha"] = matPha;
            (*result)["green"] = bgr_green;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "MattingCNN::Compute():" << e.what() << '\n';
    }
}