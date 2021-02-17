#include "samples.hpp"
#include "ns_utils.hpp"

#include <inference_engine.hpp>

using namespace InferenceEngine;

using namespace ovlib::matter;

static void InitWindows()
{
    int width = 1280;
    int height = 720;
    cv::namedWindow("com", cv::WindowFlags::WINDOW_NORMAL | cv::WindowFlags::WINDOW_FREERATIO);
    cv::resizeWindow("com", width, height);
    cv::moveWindow("com", 0, 0);
    cv::namedWindow("pha", cv::WindowFlags::WINDOW_NORMAL | cv::WindowFlags::WINDOW_FREERATIO);
    cv::resizeWindow("pha", width, height);
    cv::moveWindow("pha", 650, 0);
}

void Inference_Video()
{
#if (_MSC_VER)
    std::string model = ".\\share\\pytorch_mobilenetv2.xml";
    std::string bin = ".\\share\\pytorch_mobilenetv2.bin";
    std::string src = ".\\share\\bxg1.mp4";
    std::string bgr = ".\\share\\bxg1.png";
    std::string bgr2 = ".\\share\\replace.jpg";
#else
    std::string model = "../share/pytorch_mobilenetv2.xml";
    std::string bin = "../share/pytorch_mobilenetv2.bin";
    std::string src = "../share/src.mp4";
    std::string bgr = "../share/src.png";
    std::string bgr2 = "../share/replace.jpg";
#endif //  WINDOWS
    ovlib::matter::Shape in_shape, out_shape;
    in_shape.width = 320;
    in_shape.height = 180;
    out_shape.width = 1280;
    out_shape.height = 720;

    ovlib::matter::MatterParams params;
    ovlib::matter::MatterChannel::getDefMatterParams(params);
    params.input_shape = in_shape;
    params.path_to_model = model;
    params.path_to_bin = bin;
    params.method = ovlib::matter::METHOD_BACKGROUND_MATTING_V2;
    params.is_async = false;
    //params.effect = ovlib::matter::EFFECT_BLUR;
    params.interval = 2;
    MatterChannel* pChan = MatterChannel::create(params);
    if (!pChan)
    {
        std::cout << "Can not create Matter Channel." << std::endl;

        return;
    }

    InitWindows();

    cv::VideoCapture capture0(src);

    int framecnt = 0;
    int nDelay = 1;
    cv::Mat frame, bgrFrame, bgrFrame2;
    cv::Mat matCom, matPha;
    bgrFrame = cv::imread(bgr);
    bgrFrame2 = cv::imread(bgr2);
    std::map<std::string, ovlib::matter::FrameData> output;
    ovlib::matter::FrameData frame_com, frame_pha;
    ovlib::MatterBencher timercounter;
    timercounter.Start();
    double lElapse = 0;

    FrameData frame_bgr;
    ovlib::Utils_Ov::mat2FrameData(bgrFrame, frame_bgr);
    FrameData frame_bgr_replace;
    ovlib::Utils_Ov::mat2FrameData(bgrFrame2, frame_bgr_replace);

    while (1)
    {
        if (!capture0.isOpened())
        {
            break;
        }

        capture0 >> frame;
        if (frame.empty())
        {
            break;
        }

        framecnt++;
        {            
            ovlib::TimerCounter estimate("Phase...");
            FrameData frame_main;
            ovlib::Utils_Ov::mat2FrameData(frame, frame_main);        

            pChan->process(frame_main, frame_bgr, frame_bgr_replace, out_shape, &output);
            lElapse += timercounter.Elapse();
            std::cout << "Elapse:" << lElapse / 1000.0 << " S" << std::endl;
            delete[] frame_main.frame;
        }

        frame_com = output["com"];
        frame_pha = output["pha"];
        ovlib::Utils_Ov::frameData2Mat(frame_com, matCom);
        ovlib::Utils_Ov::frameData2Mat(frame_pha, matPha);

        cv::imshow("com", matCom);
        cv::imshow("pha", matPha);
        delete[] frame_com.frame;
        delete[] frame_pha.frame;
        char c = cv::waitKey(nDelay);
        if (c == 'c')
        {
            break;
        }
        
    }

    delete[] frame_bgr.frame;
    delete[] frame_bgr_replace.frame;

    std::cout << "Speed:" << framecnt * 1000 / (lElapse) << " FPS" << std::endl;
    capture0.release();
    cv::destroyAllWindows();
}