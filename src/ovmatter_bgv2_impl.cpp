#include "ovmatter_bgv2_impl.hpp"
#include "ns_utils.hpp"

using namespace ovlib::matter;
using namespace InferenceEngine;

REGISTER_MATTER_CLASS(METHOD_BACKGROUND_MATTING_V2, MatterBackgroundV2Impl)

MatterBackgroundV2Impl::MatterBackgroundV2Impl() : _bInit(false)
{
}

MatterBackgroundV2Impl::~MatterBackgroundV2Impl()
{
	interrupt();
	_bInit = false;
}

bool MatterBackgroundV2Impl::init(const MatterParams& param)
{
	if (_bInit)
	{
		return true;
	}

	//1 Loading Inference engine
	std::cout << "Loading Inference Engine" << std::endl;
	std::string device = param.device == "" ? "CPU" : param.device;
	Core ie;
	std::set<std::string> loadedDevices;
	std::cout << "Device info: " << device << std::endl;
	std::cout << ie.GetVersions(device) << std::endl;
	if (device.find("CPU") != std::string::npos)
	{
		ie.SetConfig({ {PluginConfigParams::KEY_DYN_BATCH_ENABLED, PluginConfigParams::YES} }, "CPU");
	}
	else if (device.find("GPU") != std::string::npos)
	{
		ie.SetConfig({ {PluginConfigParams::KEY_DYN_BATCH_ENABLED, PluginConfigParams::YES} }, "GPU");
	}
	loadedDevices.insert(device);

	//2 init CNNConfig
	if (!param.path_to_model.empty())
	{
		cv::Size shape(param.input_shape.width, param.input_shape.height);
		CNNConfig config(param.path_to_model, param.path_to_bin, shape);
		config.deviceName = device;
		config.is_async = param.is_async;
		config.scale = param.scale;
		config.max_batch_size = param.max_batch_size;
		config.cpu_bind_thread = param.cpu_bind_thread;
		config.cpu_threads_num = param.cpu_threads_num;
		config.cpu_throughput_streams = param.cpu_throughput_streams;
		config.path_to_model = param.path_to_model;
		config.path_to_bin = param.path_to_bin;
		config.effect = param.effect;
		_pCnn.reset(new CNN_Background_V2(config));

		start(); //¿ªÆôÏß³Ì

		_bInit = true;
	}
	else
	{
		_bInit = false;
	}
	return _bInit;
}

int MatterBackgroundV2Impl::process(FrameData& frame, FrameData& bgr, FrameData& bgrReplace, const ovlib::matter::Shape& shape, std::map<std::string, FrameData>* pResults)
{
	if (!_pCnn)
	{
		return -1;
	}

	if (_pCnn->getConfig()->is_async)
	{
		return doWork_async(frame, bgr, bgrReplace, shape);
	}
	else
	{
		return doWork_sync(frame, bgr, bgrReplace, shape, pResults);
	}
}

int MatterBackgroundV2Impl::doWork_async(FrameData& frame, FrameData& bgr, FrameData& bgrReplace, const ovlib::matter::Shape& shape)
{
	return -1;
}

int MatterBackgroundV2Impl::doWork_sync(FrameData& frame, FrameData& bgr, FrameData& bgrReplace, const ovlib::matter::Shape& shape, std::map<std::string, FrameData>* pResults)
{
	int ret = -1;
	if (frame.frame == 0 || bgr.frame == 0 || bgrReplace.frame == 0
		|| frame.width == 0 || frame.height == 0
		|| bgr.width == 0 || bgr.height == 0
		|| bgrReplace.width == 0 || bgrReplace.height == 0)
	{
		return ret;
	}

	if (!pResults)
	{
		return -1;
	}

	cv::Mat matFrame;
	ovlib::Utils_Ov::frameData2Mat(frame, matFrame);
	cv::Mat matBgr;
	ovlib::Utils_Ov::frameData2Mat(bgr, matBgr);
	cv::Mat matBgrReplace;
	ovlib::Utils_Ov::frameData2Mat(bgrReplace, matBgrReplace);

	_prevFrame = matFrame.clone();
	cv::Size out_shape(shape.width, shape.height);
	_pCnn->enqueue(matFrame, matBgr, matBgrReplace, out_shape);
	_pCnn->submitRequest();
	_pCnn->wait();
	_matResult = _pCnn->fetchResults();
	if (_matResult.size() <= 0)
	{
		return -1;
	}

	FrameData frameCom;
	ovlib::Utils_Ov::mat2FrameData(_matResult[0].com, frameCom);
	FrameData frameAlpha;
	ovlib::Utils_Ov::mat2FrameData(_matResult[0].pha, frameAlpha);
	(*pResults)["com"] = frameCom;
	(*pResults)["pha"] = frameAlpha;

	return 0;
}

void MatterBackgroundV2Impl::run()
{
	long lSleep = 1;
	while (!isInterrupted())
	{
		

		std::chrono::milliseconds dura(lSleep);
		std::this_thread::sleep_for(dura);
	}
}