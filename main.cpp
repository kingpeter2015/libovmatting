// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>
#include <string>

#include <onnx_import/onnx.hpp>
#include <onnx_import/onnx_utils.hpp> // onnx_import/onnx_utils.hpp provides ngraph::onnx_import::register_operator function, that registers operator in ONNX importer's set.

#include <ngraph/opsets/opset5.hpp> // ngraph/opsets/opset5.hpp provides the declaration of predefined nGraph operator set
#include <inference_engine.hpp>
//#include <details/os/os_filesystem.hpp>

#include "cnn.hpp"

using namespace InferenceEngine;

void TestMatting()
{        
}
void InitWindows()
{
    int width = 640;
    int height = 480;
    cv::namedWindow("com", cv::WindowFlags::WINDOW_NORMAL | cv::WindowFlags::WINDOW_FREERATIO);
    cv::resizeWindow("com", width, height);
    cv::moveWindow("com", 0, 0);
    cv::namedWindow("pha", cv::WindowFlags::WINDOW_NORMAL | cv::WindowFlags::WINDOW_FREERATIO);
    cv::resizeWindow("pha", width, height);
    cv::moveWindow("pha", 650, 0);
    cv::namedWindow("fgr", cv::WindowFlags::WINDOW_NORMAL | cv::WindowFlags::WINDOW_FREERATIO);
    cv::resizeWindow("fgr", width, height);
    cv::moveWindow("com", 0, 500);
}

int main(int argc, char *argv[])
{
    try
    {
        std::string model = "../share/pytorch_mobilenetv2.xml";
        std::string bin = "../share/pytorch_mobilenetv2.bin";
        //std::string model = "../pytorch_mobilenetv2.onnx";
        //std::string bin = "";
        std::string src = "../share/src.mp4";
        std::string bgr = "../share/src.png";
        cv::Size shape;
        shape.width = 480;
        shape.height = 270;
        cv::Size out_shape;
        out_shape.width = 1920;
        out_shape.height = 1080;
        CnnConfig config(model, bin, src, bgr, shape);
        config.networkCfg.nCpuThreadsNum = 0;
        config.networkCfg.nCpuThroughputStreams = 1;
        MattingCNN net(config);

        InitWindows();       
        
        cv::VideoCapture capture0(src);

        int framecnt = 0;
        int nDelay = 5;
        cv::Mat bgrFrame;
        bgrFrame = cv::imread(bgr);
        std::map<std::string,cv::Mat> output;
        cv::Mat frame,frame_com, frame_fgr, frame_pha,frame_green;
        FaceTimerCounter timercounter;
        double lElapse = 0;
        

        while (1)
        {
            if (!capture0.isOpened())
            {

                std::cout << "Video Capture Fail" << std::endl;
                break;
            }
            else
            {
                
                capture0 >> frame;
                if (frame.empty())
                {
                    std::cout << "frame.empty(): Finished" << std::endl;
                    break;
                }
                framecnt++;
                {
                    timercounter.Start();
                    TimerCounter estimate("Phase...");
                    net.Compute(frame, bgrFrame, &output, out_shape);
                    //net.Compute_Alpha(frame, bgrFrame, &output, shape);
                    lElapse += timercounter.Elapse();
                    std::cout << "Elapse:" << lElapse / 1000.0 << " S" << std::endl;
                }
                
                frame_com = output["com"];
                frame_pha = output["pha"];
                frame_fgr = output["fgr"];
                
                cv::imshow("com", frame_com);
                cv::imshow("pha", frame_pha);
                cv::imshow("fgr", frame_fgr);
                
            }
            char c = cv::waitKey(nDelay);
            if (c == 'c')
            {
                break;
            }
            
        }
        std::cout << "Speed:" << lElapse / framecnt << " FPS" << std::endl;
        capture0.release();
        cv::destroyAllWindows();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "main(int argc, char *argv[]):" << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "This sample is an API example, for any performance measurements "
                 "please use the dedicated benchmark_app tool"
              << std::endl;
    return EXIT_SUCCESS;
}
