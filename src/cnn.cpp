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
    if(_started)
    {
        return;
    }
    _elapse = 0;
    _start = std::chrono::high_resolution_clock::now();
    _started = true;
}

int64_t FaceTimerCounter::Elapse()
{
    auto ckpnt = std::chrono::high_resolution_clock::now();
    auto elapsed = ckpnt - _start;
    _elapse = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    _start = ckpnt;

    return _elapse;
}

using namespace InferenceEngine;

CnnDLSDKBase::CnnDLSDKBase(const Config &config) : _config(config) {}

void CnnDLSDKBase::Load()
{
    _cnn_network_ = _config.ie.ReadNetwork(_config.path_to_model, _config.path_to_bin);

    auto input_shapes = _cnn_network_.getInputShapes();

    const int currentBatchSize = _cnn_network_.getBatchSize();
    if (currentBatchSize != _config.max_batch_size)
        _cnn_network_.setBatchSize(_config.max_batch_size);

    _inInfo = _cnn_network_.getInputsInfo();
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
        input_dims[3] = _config._shape.width;
        input_dims[2] = _config._shape.height;
        _input_shapes[item.first] = input_dims;
    }

    _cnn_network_.reshape(_input_shapes);

    _outInfo = _cnn_network_.getOutputsInfo();
    _output_blobs_names.clear();
    for (auto &item : _outInfo)
    {
        item.second->setPrecision(Precision::FP32);
        _output_blobs_names.push_back(item.first);

        Layout layout = item.second->getLayout();
        std::cout << "Output Name:" << item.first << ", Layout Type:" << layout << std::endl;
        if(layout != Layout::NCHW)
        {
            item.second->setLayout(Layout::NC);
        }
        else
        {
            item.second->setLayout(Layout::NCHW);
        }
    }

    if (_config.networkCfg.nCpuThreadsNum > 0)
    {
        std::map<std::string, std::string> loadParams;
        loadParams[PluginConfigParams::KEY_CPU_THREADS_NUM] = std::to_string(_config.networkCfg.nCpuThreadsNum);
        loadParams[PluginConfigParams::KEY_CPU_BIND_THREAD] = _config.networkCfg.bCpuBindThread ? PluginConfigParams::YES : PluginConfigParams::NO;
        loadParams[PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS] = std::to_string(_config.networkCfg.nCpuThroughputStreams);
        _executable_network_ = _config.ie.LoadNetwork(_cnn_network_, _config.deviceName, loadParams); 
    }
    else
    {
        _executable_network_ = _config.ie.LoadNetwork(_cnn_network_, _config.deviceName);
    }

    _infer_request_ = _executable_network_.CreateInferRequest();
}

MattingCNN::MattingCNN(const Config &config) : CnnDLSDKBase(config)
{
    Load();
}

void MattingCNN::Compute2(const cv::Mat &frame, cv::Mat &bgr, cv::Mat &bgr2, std::map<std::string, cv::Mat> *result, cv::Size& outp_shape) const
{
    try
    {
        //1.Input
        cv::Mat iBgr, iFrame, matBgr, matFrame, matFrame1;
        iBgr = bgr.clone();
        iFrame = frame.clone();
        matFrame1 = frame.clone();
        unsigned char *dataMatFrame1 = matFrame1.data;

        if (frame.rows != _config._shape.height || frame.cols != _config._shape.width)
        {
            cv::resize(frame, iFrame, _config._shape);
        }

        if (bgr.rows != _config._shape.height || bgr.cols != _config._shape.width)
        {
            cv::resize(bgr, iBgr, _config._shape);
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
                size_t image_width = blob->getTensorDesc().getDims()[3];
                size_t image_height = blob->getTensorDesc().getDims()[2];
                size_t image_size = image_width * image_height;
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
        {
            TimerCounter tCount("Phase3-Outputting");
            
            //3.1 get Alpha
            Blob::Ptr blobPha = _infer_request_.GetBlob("pha");
            auto dataPha = blobPha->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();            

            size_t num_channels_pha = blobPha->getTensorDesc().getDims()[1];
            size_t image_width = blobPha->getTensorDesc().getDims()[3];
            size_t image_height = blobPha->getTensorDesc().getDims()[2];
            size_t image_size = image_width * image_height;

            cv::Mat matPha = cv::Mat::zeros(_config._shape, CV_8UC1);
            cv::Mat matCom = cv::Mat::zeros(outp_shape, CV_8UC3);
            unsigned char *dataMatPha = matPha.data;
            unsigned char *dataMatCom = matCom.data;
            for (size_t pid = 0; pid < image_size; pid++)
            {
                float alpha = dataPha[pid];
                dataMatPha[pid] = dataPha[pid] * 255.0;            
            }

            if(matPha.rows != outp_shape.height || matPha.cols != outp_shape.width)
            {
                cv::resize(matPha, matPha, outp_shape);
            }
            dataMatPha = matPha.data;
            image_size = matPha.rows * matPha.cols;

            //3.2 get bgr2
            cv::Mat matBgr2 = bgr2.clone();
            if(matBgr2.rows != outp_shape.height || matBgr2.cols != outp_shape.width)
            {
                cv::resize(matBgr2, matBgr2, outp_shape);
            }
            unsigned char *dataMatBgr2 = matBgr2.data;

            if(matFrame1.rows != outp_shape.height || matFrame1.cols != outp_shape.width)
            {
                cv::resize(matFrame1, matFrame1, outp_shape);
            }   
            dataMatFrame1 = matFrame1.data;

            //3.3 get composite
            int num_channels = matCom.channels();
            for (size_t pid = 0; pid < image_size; pid++)
            {
                int nAlpha = *(dataMatPha + pid);
                float alpha = dataMatPha[pid] / 255.0;
                int rowC = pid * num_channels;
                if(nAlpha == 0)
                {
                    *(dataMatCom + rowC + 2) = *(dataMatBgr2 + rowC + 2);
                    *(dataMatCom + rowC + 1) = *(dataMatBgr2 + rowC + 1);
                    *(dataMatCom + rowC) = *(dataMatBgr2 + rowC + 0);
                }
                else if(nAlpha == 255)
                {
                    *(dataMatCom + rowC + 2) = *(dataMatFrame1 + rowC + 2);
                    *(dataMatCom + rowC + 1) = *(dataMatFrame1 + rowC + 1);
                    *(dataMatCom + rowC) = *(dataMatFrame1 + rowC);
                }
                else
                {
                    *(dataMatCom + rowC + 2) = *(dataMatFrame1 + rowC + 2) * alpha + *(dataMatBgr2 + rowC + 2) * (1 - alpha);
                    *(dataMatCom + rowC + 1) = *(dataMatFrame1 + rowC + 1) * alpha + *(dataMatBgr2 + rowC + 1) * (1 - alpha);
                    *(dataMatCom + rowC) = *(dataMatFrame1 + rowC) * alpha + *(dataMatBgr2 + rowC) * (1 - alpha);
                    //std::cout << "alpha:" << alpha << std::endl;
                }
                /*
                if(dataMatPha[pid] == 0)
                {
                    dataMatCom[rowC + 2] = dataMatBgr2[rowC + 0];
                    dataMatCom[rowC + 1] = dataMatBgr2[rowC + 1];
                    dataMatCom[rowC + 0] = dataMatBgr2[rowC + 2];
                }
                else if(dataMatPha[pid] == 255)
                {
                    dataMatCom[rowC + 2] = dataMatFrame1[rowC + 2];
                    dataMatCom[rowC + 1] = dataMatFrame1[rowC + 1];
                    dataMatCom[rowC + 0] = dataMatFrame1[rowC + 0];
                }
                else
                {
                    dataMatCom[rowC + 2] = dataMatFrame1[rowC + 2] * alpha + dataMatBgr2[rowC + 0] * (1 - alpha);
                    dataMatCom[rowC + 1] = dataMatFrame1[rowC + 1] * alpha + dataMatBgr2[rowC + 1] * (1 - alpha);
                    dataMatCom[rowC + 0] = dataMatFrame1[rowC + 0] * alpha + dataMatBgr2[rowC + 2] * (1 - alpha);
                    //std::cout << "alpha:" << alpha << std::endl;
                }*/
                
            }
            

            (*result)["com"] = matCom;
            (*result)["pha"] = matPha;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "MattingCNN::Compute2():" << e.what() << '\n';
    }
}

void MattingCNN::Compute(const cv::Mat &frame, cv::Mat &bgr, std::map<std::string, cv::Mat> *result, cv::Size &outp_shape) const
{
    try
    {
        //1.Input
        cv::Mat iBgr, iFrame, matBgr, matFrame, matFrame1;
        iBgr = bgr.clone();
        iFrame = frame.clone();
        matFrame1 = frame.clone();
        unsigned char *dataMatFrame1 = matFrame1.data;

        if (frame.rows != _config._shape.height || frame.cols != _config._shape.width)
        {
            cv::resize(frame, iFrame, _config._shape);
        }

        if (bgr.rows != _config._shape.height || bgr.cols != _config._shape.width)
        {
            cv::resize(bgr, iBgr, _config._shape);
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
                size_t image_width = blob->getTensorDesc().getDims()[3];
                size_t image_height = blob->getTensorDesc().getDims()[2];
                size_t image_size = image_width * image_height;
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

        {
            TimerCounter tCount("Phase3-Outputting");

            //blobPha
            Blob::Ptr blobPha = _infer_request_.GetBlob("pha");
            auto dataPha = blobPha->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();

            Blob::Ptr blobFgr = _infer_request_.GetBlob("fgr");
            auto dataFgr = blobFgr->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
            size_t num_channels = blobFgr->getTensorDesc().getDims()[1];

            size_t num_channels_pha = blobPha->getTensorDesc().getDims()[1];
            size_t image_width = blobPha->getTensorDesc().getDims()[3];
            size_t image_height = blobPha->getTensorDesc().getDims()[2];
            size_t image_size = image_width * image_height;

            cv::Mat matPha = cv::Mat::zeros(_config._shape, CV_8UC1);
            cv::Mat matFgr = cv::Mat::zeros(_config._shape, CV_8UC3);
            cv::Mat matCom = cv::Mat::zeros(outp_shape, CV_8UC3);
            cv::Mat matGreen(outp_shape, CV_8UC3, cv::Scalar(120, 255, 155));
            unsigned char *dataMatPha = matPha.data;
            unsigned char *dataMatFgr = matFgr.data;
            unsigned char *dataMatCom = matCom.data;
            unsigned char *dataMatGreen = matGreen.data;
            for (size_t pid = 0; pid < image_size; pid++)
            {
                /** Iterate over all channels **/
                float alpha = dataPha[pid];
                dataMatPha[pid] = dataPha[pid] * 255.0;

                dataMatFgr[pid * num_channels + 2] = dataFgr[0 * image_size + pid] * 255.0;
                dataMatFgr[pid * num_channels + 1] = dataFgr[1 * image_size + pid] * 255.0;
                dataMatFgr[pid * num_channels + 0] = dataFgr[2 * image_size + pid] * 255.0;

                
            }

            cv::resize(matPha, matPha, outp_shape);
            cv::resize(matFgr, matFgr, outp_shape);
            dataMatPha = matPha.data;
            dataMatFgr = matFgr.data;
            image_size = matPha.rows * matPha.cols;
            /*
            cv::bitwise_and(matFrame1, matPha, matFgr);
            cv::bitwise_and(matGreen, 255-matPha, matGreen);
            cv::bitwise_xor(matFgr, matGreen, matCom);
            
            for (size_t pid = 0; pid < image_size; pid++)
            {
                char alpha = dataMatPha[pid * num_channels + 2] ? 255 : 0;
                dataMatCom[pid * num_channels + 2] = (dataMatFrame1[pid * num_channels + 2] & alpha) | (dataMatGreen[pid * num_channels + 2] & (255 - alpha));
                dataMatCom[pid * num_channels + 1] = (dataMatFrame1[pid * num_channels + 1] & alpha) | (dataMatGreen[pid * num_channels + 1] & (255 - alpha));
                dataMatCom[pid * num_channels + 0] = (dataMatFrame1[pid * num_channels + 0] & alpha) | (dataMatGreen[pid * num_channels + 0] & (255 - alpha));
            }
            */

           for (size_t pid = 0; pid < image_size; pid++)
            {
                float alpha = dataMatPha[pid] / 255.0;
                dataMatCom[pid * num_channels + 2] = (dataMatFrame1[pid * num_channels + 2] * alpha) + (dataMatGreen[pid * num_channels + 2] * (1 - alpha));
                dataMatCom[pid * num_channels + 1] = (dataMatFrame1[pid * num_channels + 1] * alpha) + (dataMatGreen[pid * num_channels + 1] * (1 - alpha));
                dataMatCom[pid * num_channels + 0] = (dataMatFrame1[pid * num_channels + 0] * alpha) + (dataMatGreen[pid * num_channels + 0] * (1 - alpha));
            }
            

            (*result)["com"] = matCom;
            (*result)["fgr"] = matFgr;
            (*result)["pha"] = matPha;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "MattingCNN::Compute():" << e.what() << '\n';
    }
}

void MattingCNN::Compute_Alpha(const cv::Mat &frame, cv::Mat &bgr, std::map<std::string, cv::Mat> *result, cv::Size &outp_shape) const
{
    cv::Mat iBgr, iFrame, matBgr, matFrame;
    iBgr = bgr.clone();
    iFrame = frame.clone();

    cv::Mat matPha = cv::Mat::zeros(frame.size(), CV_8UC3);
    cv::Mat matFgr = cv::Mat::zeros(frame.size(), CV_8UC3);
    cv::Mat matCom = cv::Mat::zeros(frame.size(), CV_8UC3);
    cv::Mat matGreen(frame.size(), CV_8UC3, cv::Scalar(120, 255, 155));
    unsigned char *dataMatPha = matPha.data;
    unsigned char *dataMatFgr = matFgr.data;
    unsigned char *dataMatCom = matCom.data;
    unsigned char *dataMatGreen = matGreen.data;

    size_t image_width = frame.size().width;
    size_t image_height = frame.size().height;
    size_t image_size = image_width * image_height;

    try
    {
        

        matBgr = iBgr.clone();
        matFrame = iFrame.clone();

        unsigned char *dataSrc = (matFrame.data);
        unsigned char *dataBgr = (matBgr.data);

        //Benchmarking Input
        {
            TimerCounter tCount("Phase1-Getting Alpha");
            cv::bitwise_xor(frame, bgr, matPha);
            cv::bitwise_and(frame, matPha, matFgr);
        }

        
        //3.Output

        {
            TimerCounter tCount("Phase3-Outputting");   

            (*result)["com"] = matCom;
            (*result)["fgr"] = matFgr;
            (*result)["pha"] = matPha;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "MattingCNN::Compute():" << e.what() << '\n';
    }
}