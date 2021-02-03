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
		std::cerr << "MatterChannel::create():" << e.what() << '\n';
		return NULL;
	}
}

int MatterChannel::getDefMatterParams(MatterParams& params)
{
	params.method = METHOD_BACKGROUND_MATTING_V2;
	params.device = "CPU";
	params.scale = 0.25;
	params.maxBatchSize = 1;
#if (LINUX)
	params.path_to_model = "./share/pytorch_mobilenetv2.xml";
	params.path_to_bin = "./share/pytorch_mobilenetv2.bin";
#else
	params.path_to_model = ".\\share\\pytorch_mobilenetv2.xml";
	params.path_to_bin = ".\\share\\pytorch_mobilenetv2.bin";
#endif // _WIN
	params.nCpuThreadsNum = 0;
	params.bCpuBindThread = true;
	params.nCpuThroughputStreams = 1;

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
