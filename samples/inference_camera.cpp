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
}

void Inference_Camera()
{
    std::string model = "../share/pytorch_mobilenetv2.xml";
    std::string bin = "../share/pytorch_mobilenetv2.bin";
    std::string src = "../share/src.mp4";
    std::string bgr = "../share/src.png";
    std::string bgr2 = "../share/replace.jpg";
    cv::Size shape;
    shape.width = 360;
    shape.height = 180;
    cv::Size out_shape;
    out_shape.width = 1280;
    out_shape.height = 720;
    CnnConfig config(model, bin, shape);
    config.networkCfg.nCpuThreadsNum = 0;
    config.networkCfg.nCpuThroughputStreams = 1;
    MattingCNN net(config);

    InitWindows();

    cv::VideoCapture capture0(src);

    int framecnt = 0;
    int nDelay = 1;
    cv::Mat bgrFrame, bgrFrame2;
    bgrFrame = cv::imread(bgr);
    bgrFrame2 = cv::imread(bgr2);
    std::map<std::string, cv::Mat> output;
    cv::Mat frame, frame_com, frame_fgr, frame_pha, frame_green;
    FaceTimerCounter timercounter;
    double lElapse = 0;

    while (1)
    {
        if (!capture0.isOpened())
            throw("Video Capture Fail\n");

        capture0 >> frame;
        if (frame.empty())
            throw("frame.empty(): Finished\n");
        
        framecnt++;
        {
            timercounter.Start();
            TimerCounter estimate("Phase...");
            net.Compute2(frame, bgrFrame, bgrFrame2, &output, out_shape);
            lElapse += timercounter.Elapse();
            std::cout << "Elapse:" << lElapse / 1000.0 << " S" << std::endl;
        }

        frame_com = output["com"];
        frame_pha = output["pha"];

        cv::imshow("com", frame_com);
        cv::imshow("pha", frame_pha);
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