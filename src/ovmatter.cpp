#include "ovmatter.h"
#include "ovmatter_base_impl.hpp"
#include "ovmatter_bgv2_impl.hpp"

using namespace ovlib::matter;

MatterChannel* MatterChannel::create(const MatterParams& params)
{
	try
	{
		MatterBaseImpl* pChannel = CMatterFactory::CreateMatter(params.method);
		if (pChannel)
		{
			pChannel->init(params);
		}

		return pChannel;
	}
	catch (const std::exception& e)
	{
		std::string msgError = e.what();
		std::cerr << "MatterChannel::create():" << msgError << '\n';
		return NULL;
	}
}

int MatterChannel::getDefMatterParams(MatterParams& params)
{
	params.method = METHOD_BACKGROUND_MATTING_V2;
	params.effect = EFFECT_NONE;
	params.device = "CPU";
	params.scale = 0.25;
	params.max_batch_size = 1;
#if (LINUX)
	params.path_to_model = "./share/pytorch_mobilenetv2.xml";
	params.path_to_bin = "./share/pytorch_mobilenetv2.bin";
#else
	params.path_to_model = ".\\share\\pytorch_mobilenetv2.xml";
	params.path_to_bin = ".\\share\\pytorch_mobilenetv2.bin";
#endif // _WIN
	params.cpu_threads_num = 0;
	params.cpu_bind_thread = true;
	params.cpu_throughput_streams = 1;

	params.input_shape.width = 256;
	params.input_shape.height = 144;

	params.is_async = false;
	params.interval = 1;
	params.threshold_motion = -0.1f;

	return 0;	
}

void MatterChannel::destroyed(MatterChannel* pChan)
{
	std::cout << "MatterChannel::destroy " << pChan << std::endl;
	if (pChan)
	{
		MatterChannel* tmp = pChan;
		pChan = nullptr;
		delete tmp;
	}
}
