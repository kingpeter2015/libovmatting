/*
 * ovmatter.h
 *
 */

#ifndef __OV_MATTER__H__
#define __OV_MATTER__H__

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <utility>
#include <iostream>

#ifdef OVMATTING_EXPORTS
#define OV_MATTER_API __declspec(dllexport)
#else
#define OV_MATTER_API
#endif

#define OV_MATTER_VERSION "0.0.1"

namespace ovlib 
{
    namespace matter
    {
        //Methods of Background matting
        enum OV_MATTER_API MattingMethod
        {
            METHOD_BACKGROUND_MATTING_V2 = 0x00,
            METHOD_MODNET = 0x01,
        };

        enum OV_MATTER_API MATTER_EFFECT
        {
            EFFECT_NONE,
            EFFECT_BLUR
        };

        struct OV_MATTER_API Rect
        {
            int left;
            int top;
            int right;
            int bottom;
        };

        struct OV_MATTER_API Shape
        {
            unsigned int width;
            unsigned int height;
        };

        enum OV_MATTER_API FRAME_FORMAT
        {
            FRAME_FOMAT_ERROR = 0x00,
            FRAME_FOMAT_I420 = 0x01,
            FRAME_FOMAT_BGR = 0x02,
            FRAME_FOMAT_RGB = 0x03,
            FRAME_FOMAT_GRAY = 0x04
        };

        struct OV_MATTER_API FrameData
        {
            unsigned char* frame=0;            
            unsigned int height=0;
            unsigned int width=0;
            FRAME_FORMAT format= FRAME_FOMAT_ERROR;
            void* tag=0;
        };

        const FrameData NullFrame;

        struct OV_MATTER_API MatterParams
        {       
            MattingMethod method;
            MATTER_EFFECT effect;

            std::string device; 
            std::string path_to_model;
            std::string path_to_bin;
            
            float scale;            
            int max_batch_size;

            int cpu_threads_num;    //default 0
            bool cpu_bind_thread; //default true
            int cpu_throughput_streams;    //default 1

            bool is_async;
            int interval;
            float threshold_motion;

            Shape input_shape;

        };        

        class OV_MATTER_API MatterChannel
        {
        public:
            static MatterChannel* create(const MatterParams& params);
            static int getDefMatterParams(MatterParams& params);
            static void destroyed(MatterChannel* pChan);

            virtual ~MatterChannel() {};
            
            virtual int process(FrameData& frame, FrameData& bgr, FrameData& bgrReplace, const ovlib::matter::Shape& out_shape, std::map<std::string, FrameData>* pResults = 0) = 0;
            virtual int getInferCount() = 0;

            virtual void setStrategy_async(bool bAuto = true, int interval = 0, const Shape& input_shape = { 0,0 }, const Shape& out_shape = {0,0}) = 0;
            virtual void setBackground_async(FrameData& bgrReplace, MATTER_EFFECT = EFFECT_NONE, const FrameData& bgr = NullFrame) = 0;
            virtual int process_async(FrameData& frame, FrameData& frameCom, FrameData& frameAlpha, const ovlib::matter::Shape& out_shape = { 0,0 }) = 0;
        };
    }; // namespace matter
}; // namespace ovlib

#endif
