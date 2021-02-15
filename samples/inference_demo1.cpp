#include "samples.hpp"
#include "ns_utils.hpp"

#include <inference_engine.hpp>

using namespace InferenceEngine;

using namespace ovlib::matter;

static void showUsage() 
{
    std::cout << std::endl;
    std::cout << "ovmatter [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                            " << "Print a usage message." << std::endl;
    std::cout << "    -src     '<path>'             " << "Required. Path to a video or image file" << std::endl;
    std::cout << "    -src_bgr '<path>'             " << "Path to source video background image file." << std::endl;
    std::cout << "    -dst     '<path>'             " << "Required. Path to destination video or image file" << std::endl;
    std::cout << "    -dst_bgr '<path>'             " << "Path to destination background image file for replacing." << std::endl;
    std::cout << "    -model   '<path>'             " << "Required. Path to the model (.xml) file." << std::endl;
    std::cout << "    -dev     '<device>'           " << "Optional.Specify the target device for Inference System (the list of available devices is shown below).Default value is CPU. The application looks for a suitable plugin for the specified device." << std::endl;
    std::cout << "    -bin                          " << "Required. Path to the model (.bin) file." << std::endl;
    std::cout << "    -in_width                     " << "Optional. Width of input shape. Default: 320" << std::endl;
    std::cout << "    -in_height                    " << "Optional. Height of input shape. Default: 180" << std::endl;
    std::cout << "    -method                       " << "Required. Matting Method, 0-background v2;1-modnet" << std::endl;
}

void Inference_demo1(int argc, char* argv[])
{
    //1.get matter parameters
    ovlib::matter::MatterParams params;
    ovlib::matter::MatterChannel::getDefMatterParams(params);

#if (_MSC_VER)
    std::string model = ".\\share\\pytorch_mobilenetv2.xml";
    std::string bin = ".\\share\\pytorch_mobilenetv2.bin";
    std::string src = ".\\share\\src.mp4";
    std::string dst = ".\\share\\dst.mp4";
    std::string bgr = ".\\share\\src.png";
    std::string bgr2 = ".\\share\\replace.jpg";
#else
    std::string model = "../share/pytorch_mobilenetv2.xml";
    std::string bin = "../share/pytorch_mobilenetv2.bin";
    std::string src = "../share/src.mp4";
    std::string dst = "../share/dst.mp4";
    std::string bgr = "../share/src.png";
    std::string bgr2 = "../share/replace.jpg";
#endif //  WINDOWS
    ovlib::matter::Shape in_shape, out_shape;
    in_shape.width = 320;
    in_shape.height = 180;
    out_shape.width = 1280;
    out_shape.height = 720;

    for (int i = 1; i < argc; i++)
    {
        const char* pc = argv[i];
        if (pc[0] == '-' && pc[1])
        {
            if (!::strncmp(pc, "-dev", 4))
            {
                params.device = argv[++i];
            }
            else if (!::strncmp(pc, "-model", 6))
            {
                params.path_to_model = argv[++i];
            }
            else if (!::strncmp(pc, "-bin", 4))
            {
                params.path_to_bin = argv[++i];
            }
            else if (!::strncmp(pc, "-src", 4))
            {
                src = argv[++i];
            }
            else if (!::strncmp(pc, "-src_bgr", 8))
            {
                bgr = argv[++i];
            }
            else if (!::strncmp(pc, "-dst_bgr", 8))
            {
                bgr2 = argv[++i];
            }
            else if (!::strncmp(pc, "-dst", 4))
            {
                dst = argv[++i];
            }
            else if (!::strncmp(pc, "-in_width", 9))
            {
                in_shape.width = atoi(argv[++i]);
            }
            else if (!::strncmp(pc, "-in_height", 10))
            {
                in_shape.width = atoi(argv[++i]);
            }
            else if (!::strncmp(pc, "-method", 7))
            {
                int nMethod = atoi(argv[++i]);
                if (nMethod == 0)
                {
                    params.method = ovlib::matter::METHOD_BACKGROUND_MATTING_V2;
                }
                else
                {
                    params.method = ovlib::matter::METHOD_MODNET;
                }
            }
            else if (!::strncmp(pc, "-h", 2))
            {
                showUsage();
                return;
            }
        }
    }

    
    params.input_shape = in_shape;
    params.path_to_model = model;
    params.path_to_bin = bin;
    //params.method = ovlib::matter::METHOD_MODNET;
    params.is_async = false;
    //params.effect = ovlib::matter::EFFECT_BLUR;

    if (params.method == ovlib::matter::METHOD_MODNET)
    {
        params.path_to_model = ".\\share\\modnet.xml";
        params.path_to_bin = ".\\share\\modnet.bin";
        params.input_shape.width = 512;
        params.input_shape.height = 512;
    }
    MatterChannel* pChan = MatterChannel::create(params);
    if (!pChan)
    {
        std::cout << "Can not create Matter Channel." << std::endl;

        return;
    }

    //2. src capture and dst writer
    cv::VideoCapture capture0(src);
    out_shape.width = capture0.get(cv::CAP_PROP_FRAME_WIDTH);
    out_shape.height = capture0.get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::VideoWriter writer;
    bool bRet = writer.open(dst, cv::VideoWriter::fourcc('X', '2', '6', '4'), 60, cv::Size(out_shape.width, out_shape.height));
    if (!bRet)
    {
        std::cout << "cv::VideoWriter Initializes failed!" << std::endl;
    }

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

    //3.write 
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
        writer.write(matCom);

        delete[] frame_com.frame;
        delete[] frame_pha.frame;
        
    }
    writer.release();

    delete[] frame_bgr.frame;
    delete[] frame_bgr_replace.frame;

    std::cout << "Speed:" << framecnt * 1000 / (lElapse) << " FPS" << std::endl; 

    MatterChannel::destroyed(pChan);
    
    capture0.release();
    
}