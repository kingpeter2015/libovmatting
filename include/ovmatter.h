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
            METHOD_BACKGROUND_MATTING_V2 = 0x00
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
            FRAME_FOMAT_I420 = 0x00,
            FRAME_FOMAT_BGR = 0x01,
            FRAME_FOMAT_RGB = 0x02,
            FRAME_FOMAT_GRAY = 0x03
        };

        struct OV_MATTER_API FrameData
        {
            unsigned char* frame;            
            unsigned int height;
            unsigned int width;
            FRAME_FORMAT format;
            void* tag;
        };

        struct OV_MATTER_API MatterParams
        {       
            MattingMethod method;

            std::string device; 
            std::string path_to_model;
            std::string path_to_bin;
            
            float scale;            
            int max_batch_size;

            int cpu_threads_num;    //default 0
            bool cpu_bind_thread; //default true
            int cpu_throughput_streams;    //default 1
            bool is_async;

            Shape input_shape;
        };

        class OV_MATTER_API MatterChannel
        {
        public:
            static MatterChannel* create(const MatterParams& params);
            static int getDefMatterParams(MatterParams& params);
            static void destroyed(MatterChannel* pChan);

            virtual ~MatterChannel() {};
            virtual int process(FrameData& frame, FrameData& bgr, FrameData& bgrReplace, std::map<std::string, FrameData>& results, const ovlib::matter::Shape& shape) = 0;
        };
    }; // namespace matter
}; // namespace ovlib

#endif
