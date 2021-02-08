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
#include "ns_thread.hpp"
#include "ns_utils.hpp"

namespace ovlib
{
	namespace matter
	{
		class MatterBackgroundV2Impl : public MatterBaseImpl, public ov_simple_thread
		{
		public:
			MatterBackgroundV2Impl();
			virtual ~MatterBackgroundV2Impl();

			/***for MatterBaseImpl*****/
			virtual bool init(const MatterParams& params) override;
			virtual int process(FrameData& frame, FrameData& bgr, FrameData& bgrReplace, const ovlib::matter::Shape& shape, std::map<std::string, FrameData>* pResults = 0) override;
			virtual void setStrategy_async(bool bAuto = true, int interval = 0, const Shape& input_shape = { 0,0 }, const Shape& out_shape = { 0,0 }) override;
			virtual void setBackground_async(FrameData& bgrReplace, MATTER_EFFECT = EFFECT_NONE, const FrameData& bgr = { 0, 0, 0, FRAME_FOMAT_BGR, 0 }) override;
			virtual int process_async(FrameData& frame, FrameData& frameCom, FrameData& frameAlpha, const ovlib::matter::Shape& out_shape) override;

			/***for ov_simple_thread***/
			virtual void run() override;

		private:
			int doWork_sync(FrameData& frame, FrameData& bgr, FrameData& bgrReplace, const ovlib::matter::Shape& shape, std::map<std::string, FrameData>* pResults);
			int doWork_sync_V2(cv::Mat& frame, cv::Mat& bgr, cv::Mat& bgrReplace, cv::Size& out_shape, cv::Mat& matCom, cv::Mat& matAlpha);
			void compose(cv::Mat& src, cv::Mat& replace, cv::Mat& alpha, cv::Mat& com, cv::Size& out_shape);

		private:			
			std::unique_ptr<CNN_Background_V2> _pCnn;

			bool _bInit;

			cv::Mat _prevFrame;
			MattingObjects _matResult;

		private: /****for asynchronous*****/
			ovlib::MatterBencher _bencher;
			long long _elapse;
			mutable std::mutex mutex_;
			ConcurrentQueue<cv::Mat> _queue_input;
			ConcurrentQueue<MattingObject> _queue_output;
			cv::Mat _frame_bgr;
			cv::Mat _frame_replace;
			MATTER_EFFECT _effect;
			cv::Size _shape_input;
			cv::Size _shape_output;
			int _interval;

		};
	}; // namespace matter
}; // namespace ovlib


#endif