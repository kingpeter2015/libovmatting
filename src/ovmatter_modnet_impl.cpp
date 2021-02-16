#include "ovmatter_modnet_impl.hpp"
#include "ns_utils.hpp"

#if (_MSC_VER)
#include <direct.h>
#else
#include <unistd.h>
#endif

using namespace ovlib::matter;
using namespace InferenceEngine;

REGISTER_MATTER_CLASS(METHOD_MODNET, MatterModnetImpl)

MatterModnetImpl::MatterModnetImpl() : _bInit(false), _elapse(0), _shape_output(cv::Size(1280,720))
{
	m_nInterval = 1;
}

MatterModnetImpl::~MatterModnetImpl()
{
	interrupt();
	_bInit = false;
}

bool MatterModnetImpl::init(const MatterParams& param)
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
		
		std::string sModelXml, sModelBin, sCwdPath;
#if (_MSC_VER)
		sCwdPath = _getcwd(NULL, 0);
#else
		sCwdPath = getcwd(NULL, 0);
#endif
		std::cout << "current working path:" << sCwdPath << std::endl;

		sModelXml = Utils_Ov::getRealPath(param.path_to_model);
		sModelBin = Utils_Ov::getRealPath(param.path_to_bin);
		std::cout << "xml path:" << sModelXml << ", bin path:" << sModelBin << std::endl;

		CNNConfig config(sModelXml, sModelBin, shape);
		config.deviceName = device;
		config.is_async = param.is_async;
		config.scale = param.scale;
		config.max_batch_size = param.max_batch_size;
		config.cpu_bind_thread = param.cpu_bind_thread;
		config.cpu_threads_num = param.cpu_threads_num;
		config.cpu_throughput_streams = param.cpu_throughput_streams;
		config.path_to_model = param.path_to_model;
		config.path_to_bin = param.path_to_bin;
		config.interval = param.interval;
		config.motion_threshold = param.threshold_motion;
		m_nInterval = param.interval;
		m_fMotionThreshold = param.threshold_motion;
		_pCnn.reset(new CNN_Modnet(config));

		start(); //�����߳�

		_bInit = true;
	}
	else
	{
		_bInit = false;
	}
	return _bInit;
}

int MatterModnetImpl::process(FrameData& frame, FrameData& bgr, FrameData& bgrReplace, const ovlib::matter::Shape& shape, std::map<std::string, FrameData>* pResults)
{
	if (!_pCnn)
	{
		return -1;
	}
	if (!_pCnn->getAsync())
	{
		_pCnn->setAsync(true);
	}

	return doWork_sync(frame, bgr, bgrReplace, shape, pResults);
}

/// <summary>
/// Infer Synchronously
/// </summary>
/// <param name="frame"></param>
/// <param name="bgr"></param>
/// <param name="bgrReplace"></param>
/// <param name="shape"></param>
/// <param name="pResults"></param>
/// <returns></returns>
int MatterModnetImpl::doWork_sync(FrameData& frame, FrameData& bgr, FrameData& bgrReplace, const ovlib::matter::Shape& shape, std::map<std::string, FrameData>* pResults)
{
	int ret = -1;
	if (frame.frame == 0 || bgrReplace.frame == 0
		|| frame.width == 0 || frame.height == 0
		|| bgrReplace.width == 0 || bgrReplace.height == 0)
	{
		return ret;
	}

	if (!pResults)
	{
		return -1;
	}

	static int l_frame_count_sync = 0;
	l_frame_count_sync++;


	cv::Mat matFrame;
	ovlib::Utils_Ov::frameData2Mat(frame, matFrame);
	cv::Mat matBgr;
	ovlib::Utils_Ov::frameData2Mat(bgr, matBgr);
	cv::Mat matBgrReplace;
	ovlib::Utils_Ov::frameData2Mat(bgrReplace, matBgrReplace);
	cv::Size out_shape(shape.width, shape.height);
	cv::Mat matCom;
	cv::Mat matPha;

	//1.check if frame changes
	if (m_fMotionThreshold > 0.0f)
	{
		double preDiff;
		double dblDiff = Utils_Ov::getSceneScore(_prevFrame, matFrame, preDiff);
		bool bExist = (_preResult.find("pha") != _preResult.end());
		if (dblDiff < m_fMotionThreshold && bExist)
		{
			matPha = _preResult["pha"];
			compose(matFrame, matBgrReplace, matPha, matCom, out_shape);
			FrameData frameCom;
			ovlib::Utils_Ov::mat2FrameData(matCom, frameCom);
			FrameData frameAlpha;
			ovlib::Utils_Ov::mat2FrameData(matPha, frameAlpha);
			(*pResults)["com"] = frameCom;
			(*pResults)["pha"] = frameAlpha;
			return 0;
		}
	}
	_prevFrame = matFrame.clone();

	//2 skip Frame intervally 
	if (m_nInterval <= 0)
	{
		m_nInterval = 1;
	}
	l_frame_count_sync = l_frame_count_sync % m_nInterval;
	bool bExist = (_preResult.find("pha") != _preResult.end());
	if (l_frame_count_sync != 0 && bExist)
	{
		matPha = _preResult["pha"];
		compose(matFrame, matBgrReplace, matPha, matCom, out_shape);
		FrameData frameCom;
		ovlib::Utils_Ov::mat2FrameData(matCom, frameCom);
		FrameData frameAlpha;
		ovlib::Utils_Ov::mat2FrameData(matPha, frameAlpha);
		(*pResults)["com"] = frameCom;
		(*pResults)["pha"] = frameAlpha;
		return 0;
	}

	//3. Infer result
	{
		_pCnn->enqueue("input.1", matFrame);
		_pCnn->submitRequest();
		_pCnn->wait();
		_matResult = _pCnn->fetchResults();
		m_nInferCount++;
		if (_matResult.size() <= 0)
		{
			return -1;
		}

		matPha = _matResult[0].pha;
		
		compose(matFrame, matBgrReplace, matPha, matCom, out_shape);

		FrameData frameCom;
		ovlib::Utils_Ov::mat2FrameData(matCom, frameCom);
		FrameData frameAlpha;
		ovlib::Utils_Ov::mat2FrameData(_matResult[0].pha, frameAlpha);
		(*pResults)["com"] = frameCom;
		(*pResults)["pha"] = frameAlpha;

		_preResult["com"] = matCom;
		_preResult["pha"] = _matResult[0].pha;
	}

	return 0;
}

void MatterModnetImpl::compose(cv::Mat& src, cv::Mat& replace, cv::Mat& alpha, cv::Mat& com, cv::Size& out_shape)
{
	
	if (out_shape.height == 0 || out_shape.width == 0)
	{
		out_shape.width = 1280;
		out_shape.height = 720;
	}
	cv::Size l_shape = out_shape;
	com = cv::Mat::zeros(l_shape, CV_8UC3);
	cv::Mat matSrc, matReplace;
	matSrc = src.clone();
	matReplace = replace.clone();
	if (matSrc.rows != l_shape.height || matSrc.cols != l_shape.width)
	{
		cv::resize(matSrc, matSrc, l_shape);
	}
	if (matReplace.rows != l_shape.height || matReplace.cols != l_shape.width)
	{
		cv::resize(matReplace, matReplace, l_shape);
	}
	if (alpha.rows != l_shape.height || alpha.cols != l_shape.width)
	{
		cv::resize(alpha, alpha, l_shape, 0, 0, cv::INTER_CUBIC);
	}
	unsigned char* dataMatPha = alpha.data;
	unsigned char* dataMatCom = com.data;
	unsigned char* dataMatFrame1 = matSrc.data;
	unsigned char* dataMatBgr2 = matReplace.data;
	int num_channels = 3;
	int image_size = alpha.rows * alpha.cols;
	for (size_t pid = 0; pid < image_size; pid++)
	{
		int nAlpha = *(dataMatPha + pid);
		int rowC = pid * num_channels;
		if (nAlpha == 0)
		{
			*(dataMatCom + rowC + 2) = *(dataMatBgr2 + rowC + 2);
			*(dataMatCom + rowC + 1) = *(dataMatBgr2 + rowC + 1);
			*(dataMatCom + rowC) = *(dataMatBgr2 + rowC);
		}
		else if (nAlpha == 255)
		{
			*(dataMatCom + rowC + 2) = *(dataMatFrame1 + rowC + 2);
			*(dataMatCom + rowC + 1) = *(dataMatFrame1 + rowC + 1);
			*(dataMatCom + rowC) = *(dataMatFrame1 + rowC);
		}
		else
		{
			float falpha = nAlpha / 255.0;
			*(dataMatCom + rowC + 2) = *(dataMatFrame1 + rowC + 2) * falpha + *(dataMatBgr2 + rowC + 2) * (1 - falpha);
			*(dataMatCom + rowC + 1) = *(dataMatFrame1 + rowC + 1) * falpha + *(dataMatBgr2 + rowC + 1) * (1 - falpha);
			*(dataMatCom + rowC) = *(dataMatFrame1 + rowC) * falpha + *(dataMatBgr2 + rowC) * (1 - falpha);
		}
	}	
}

/****************************************Asynchronous Process*******************************************************/
void MatterModnetImpl::setStrategy_async(bool bAuto, int interval, const Shape& input_shape, const Shape& out_shape)
{

}

void MatterModnetImpl::setBackground_async(FrameData& bgrReplace, MATTER_EFFECT effect, const FrameData& bgr)
{	
	cv::Mat matReplace, matBgr;
	ovlib::Utils_Ov::frameData2Mat(bgrReplace, matReplace);
	FrameData bgrFd = bgr;
	ovlib::Utils_Ov::frameData2Mat(bgrFd, matBgr);

	std::lock_guard<std::mutex> lock(mutex_);
	if (!matReplace.empty())
	{
		_frame_replace = matReplace.clone();
	}
	if (!matBgr.empty())
	{
		_frame_bgr = matBgr.clone();
	}

	if (effect == ovlib::matter::EFFECT_BLUR)
	{
		//cv::bilateralFilter(_frame_replace, _frame_replace, 30, 500.0, 10.0);
		cv::GaussianBlur(_frame_replace, _frame_replace, cv::Size(3, 3), 11.0, 11.0);
	}
}

int MatterModnetImpl::process_async(FrameData& frame, FrameData& frameCom, FrameData& frameAlpha, const ovlib::matter::Shape& out_shape)
{
	static int l_frame_count = 0;
	//1.put input frame into input_queue 
	if (frame.frame == 0 || frame.height == 0 || frame.width == 0)
	{
		//do nothing
	}
	else
	{
		l_frame_count++;
		if (m_nInterval <= 0)
		{
			m_nInterval = 1;
		}
		l_frame_count = l_frame_count % m_nInterval;
		if (l_frame_count == 0)
		{
			cv::Mat matFrame;
			Utils_Ov::frameData2Mat(frame, matFrame);
			_queue_input.push(matFrame);
		}
	}

	//1.2 input shape
	if (out_shape.height > 0 && out_shape.width > 0)
	{
		_shape_output.width = out_shape.width;
		_shape_output.height = out_shape.height;
	}

	// 2.return raw frame if out put shape is empty or replace bgr is empty
	if (_shape_output.empty() || _frame_replace.empty())
	{
		frameCom = frame;
		return -1;
	}
	
	if (_frame_replace.rows != _shape_output.height || _frame_replace.cols != _shape_output.width)
	{
		cv::resize(_frame_replace, _frame_replace, _shape_output);
	}

	// 3.if output queue is empty, return replace bgr
	if (_queue_output.size() <= 0)
	{
		Utils_Ov::mat2FrameData(_frame_replace, frameCom);
		frameAlpha.frame = 0;
		return 1;	
	}
	/*
	else if (_queue_output.size() == 1)
	{
		MattingObject f;
		_queue_output.front(f);
		Utils_Ov::mat2FrameData(f.com, frameCom);
		Utils_Ov::mat2FrameData(f.pha, frameAlpha);
		return 1;
	}*/

	while (_queue_output.size() > 1)
	{
		MattingObject f;
		_queue_output.pop(f);
	}
	// 4. if output queue is not empty, return first item in the output queue
	MattingObject result;
	_queue_output.front(result);
	Utils_Ov::mat2FrameData(result.com, frameCom);
	Utils_Ov::mat2FrameData(result.pha, frameAlpha);

	return 0;
}

void MatterModnetImpl::run()
{
	long lSleep = 5;
	while (!isInterrupted())
	{
		Utils_Ov::sleep(lSleep);
		if (_queue_input.size() <= 0)
		{			
			continue;
		}

		cv::Mat frame, matAlpha, matCom;
		_queue_input.pop(frame);
		{
			std::lock_guard<std::mutex> lock(mutex_);
			_bencher.Start();
			if (!_pCnn->getAsync())
			{
				_pCnn->setAsync(true);
			}

			_pCnn->enqueue("input.1", frame);
			_pCnn->submitRequest();
			_pCnn->wait();
			_matResult = _pCnn->fetchResults();
			if (_matResult.size() <= 0)
			{
				continue;
			}

			matAlpha = _matResult[0].pha;
			compose(frame, _frame_replace, matAlpha, matCom, _shape_output);
			if (matAlpha.empty() || matCom.empty())
			{
				continue;
			}
			MattingObject obj;
			obj.com = matCom.clone();
			obj.pha = matAlpha.clone();
			_queue_output.push(obj);
			_elapse = _bencher.Elapse();
		}

	}
}