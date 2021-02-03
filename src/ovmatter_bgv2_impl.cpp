#include "ovmatter_bgv2_impl.hpp"

using namespace ovlib::matter;

MatterBackgroundV2Impl::MatterBackgroundV2Impl() : _bInit(false)
{
}

MatterBackgroundV2Impl::~MatterBackgroundV2Impl()
{

}

bool MatterBackgroundV2Impl::init(const MatterParams& param)
{
	if (_bInit)
	{
		return true;
	}

	CNNConfig config(param.path_to_model, param.path_to_bin)
}
