#include "samples.hpp"

static void InitWindows()
{
    int width = 640;
    int height = 480;
    cv::namedWindow("com", cv::WindowFlags::WINDOW_NORMAL | cv::WindowFlags::WINDOW_FREERATIO);
    cv::resizeWindow("com", width, height);
    cv::moveWindow("com", 0, 0);
    cv::namedWindow("pha", cv::WindowFlags::WINDOW_NORMAL | cv::WindowFlags::WINDOW_FREERATIO);
    cv::resizeWindow("pha", width, height);
    cv::moveWindow("pha", 650, 0);
    //cv::namedWindow("fgr", cv::WindowFlags::WINDOW_NORMAL | cv::WindowFlags::WINDOW_FREERATIO);
    //cv::resizeWindow("fgr", width, height);
    //cv::moveWindow("com", 0, 500);
}

void Inference_Video()
{
    std::string model = "../share/pytorch_mobilenetv2.xml";
    std::string bin = "../share/pytorch_mobilenetv2.bin";
    std::string src = "../share/src.mp4";
    std::string bgr = "../share/src.png";
    cv::Size shape;
    shape.width = 360;
    shape.height = 180;
    cv::Size out_shape;
    out_shape.width = 1920;
    out_shape.height = 1080;
    CnnConfig config(model, bin, shape);
    config.networkCfg.nCpuThreadsNum = 0;
    config.networkCfg.nCpuThroughputStreams = 1;
    MattingCNN net(config);

    InitWindows();

    cv::VideoCapture capture0(src);

    int framecnt = 0;
    int nDelay = 5;
    cv::Mat bgrFrame;
    bgrFrame = cv::imread(bgr);
    std::map<std::string, cv::Mat> output;
    cv::Mat frame, frame_com, frame_fgr, frame_pha, frame_green;
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

            cv::imshow("com", frame_com);
            cv::imshow("pha", frame_pha);
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