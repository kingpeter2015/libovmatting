#ifndef __OVMATTER__BACKGROUND_MATTING_V2_IMPL_HPP__
#define __OVMATTER__BACKGROUND_MATTING_V2_IMPL_HPP__

/**
* MIT License

Copyright (c) 2021 kingpeter2015

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

* Please Ref: https://github.com/PeterL1n/BackgroundMattingV2
**/

#include "ovmatter.h"
#include "ovmatter_base_impl.hpp"
#include "cnn_backgroundv2.hpp"

namespace ovlib
{
	namespace matter
	{
		class MatterBackgroundV2Impl : public MatterBaseImpl
		{
		public:
			MatterBackgroundV2Impl();
			virtual ~MatterBackgroundV2Impl();
			virtual bool init(const MatterParams& params);
			virtual int process(const std::vector<CFrameData>& images, std::map<std::string, CFrameData>& results);
		private:
			std::unique_ptr<CNN_Background_V2> _pCnn;
			std::unique_ptr<CNNConfig> _pConfig;
			bool _bInit;
		};
	}; // namespace matter
}; // namespace ovlib


#endif