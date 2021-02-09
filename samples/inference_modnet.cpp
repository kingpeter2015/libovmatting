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

void Inference_Modnet()
{
#if (_MSC_VER)
	std::string model = ".\\share\\modnet.xml";
	std::string bin = ".\\share\\modnet.bin";
	std::string src = ".\\share\\src.mp4";
	std::string bgr = ".\\share\\src.png";
	std::string bgr2 = ".\\share\\replace.jpg";
#else
	std::string model = "../share/pytorch_mobilenetv2.xml";
	std::string bin = "../share/pytorch_mobilenetv2.bin";
	std::string src = "../share/src.mp4";
	std::string bgr = "../share/src.png";
	std::string bgr2 = "../share/replace.jpg";
#endif //  WINDOWS
	ovlib::matter::Shape in_shape, out_shape;
	in_shape.width = 256;
	in_shape.height = 256;
	out_shape.width = 1280;
	out_shape.height = 720;

	ovlib::matter::MatterParams params;
	ovlib::matter::MatterChannel::getDefMatterParams(params);
	params.input_shape = in_shape;
	params.path_to_model = model;
	params.path_to_bin = bin;
	params.method = ovlib::matter::METHOD_MODNET;
	params.is_async = false;
	//params.effect = ovlib::matter::EFFECT_BLUR;
	MatterChannel* pChan = MatterChannel::create(params);
	if (!pChan)
	{
		std::cout << "Can not create Matter Channel." << std::endl;

		return;
	}

	InitWindows();

	cv::VideoCapture capture0(src);

	int framecnt = 0;
	int nDelay = 5;
	cv::Mat frame, bgrFrame, bgrFrame2;
	
	bgrFrame = cv::imread(bgr);
	bgrFrame2 = cv::imread(bgr2);
	std::map<std::string, ovlib::matter::FrameData> output;
	ovlib::matter::FrameData frame_com, frame_pha;
	ovlib::MatterBencher timercounter;
	timercounter.Start();
	double lElapse = 0;

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

			FrameData frame_main, frame_bgr, frame_bgr_replace;
			frame_main.format = FRAME_FOMAT_BGR;
			frame_main.width = frame.cols;
			frame_main.height = frame.rows;
			frame_main.frame = frame.data;
			frame_bgr_replace.format = FRAME_FOMAT_BGR;
			frame_bgr_replace.width = bgrFrame2.cols;
			frame_bgr_replace.height = bgrFrame2.rows;
			frame_bgr_replace.frame = bgrFrame2.data;


			pChan->process(frame_main, frame_bgr, frame_bgr_replace, out_shape, &output);
			lElapse += timercounter.Elapse();
			std::cout << "Elapse:" << lElapse / 1000.0 << " S" << std::endl;
		}

		frame_com = output["com"];
		frame_pha = output["pha"];
		cv::Mat matCom, matPha;
		ovlib::Utils_Ov::frameData2Mat(frame_com, matCom);
		ovlib::Utils_Ov::frameData2Mat(frame_pha, matPha);

		cv::imshow("com", matCom);
		cv::imshow("pha", matPha);
		char c = cv::waitKey(nDelay);
		if (c == 'c')
		{
			break;
		}
	}
	std::cout << "Speed:" << framecnt * 1000 / (lElapse) << " FPS" << std::endl;
	capture0.release();
	cv::destroyAllWindows();
}