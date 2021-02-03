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

        struct OV_MATTER_API CRect
        {
            int left;
            int top;
            int right;
            int bottom;
        };

        struct OV_MATTER_API CFrameData
        {
            unsigned char* pFrame;            
            int height;
            int width;
            int channel;
            int type;
            void* tag;
        };

        struct OV_MATTER_API MatterParams
        {       
            MattingMethod method;

            std::string device; 
            std::string path_to_model;
            std::string path_to_bin;
            
            float scale;            
            int maxBatchSize;

            int nCpuThreadsNum;    //default 0
            bool bCpuBindThread; //default true
            int nCpuThroughputStreams;    //default 1
        };

        class OV_MATTER_API MatterChannel
        {
        public:
            static MatterChannel* create(const MatterParams& params);
            static int getDefMatterParams(MatterParams& params);
            static void destroyed(MatterChannel* pChan);

            virtual ~MatterChannel() {};
            virtual int process(const std::vector<CFrameData>& images, std::map<std::string,CFrameData>& results) = 0;
        };
    }; // namespace matter
}; // namespace ovlib

#endif
