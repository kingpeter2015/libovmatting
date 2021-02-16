#ifndef __NS__UTILS__HPP___
#define __NS__UTILS__HPP___
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>
#include <memory>

#include <functional>
#include <chrono>
#include <time.h>

#include "ovmatter.h"

namespace ovlib
{
    class OV_MATTER_API Utils_String
    {
    public:
        static std::vector<std::string> split(const std::string &s, char delim);
    };


    class OV_MATTER_API TimerCounter
    {
    public:
        TimerCounter(std::string name)
        {
            _name = name;
            _start = std::chrono::high_resolution_clock::now();
        }
        ~TimerCounter()
        {
            auto elapsed = std::chrono::high_resolution_clock::now() - _start;
#if (_MSC_VER)
            _elapse = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();            
#else
            _elapse = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
#endif // _WIN


            std::cout << "Benchmarking " << _name << "\t\t| elapse miliseconds: " << _elapse / 1.0 << ",\t estimate fps: " << 1000.0 / _elapse << std::endl;
        }
    private:
        std::string _name;
#if (_MSC_VER)
        std::chrono::time_point<std::chrono::steady_clock> _start;
#else
        std::chrono::_V2::system_clock::time_point _start;
#endif // _WIN

        int64_t _elapse;
        bool _started = false;
    };

    class OV_MATTER_API MatterBencher
    {
    public:
        MatterBencher();
        virtual ~MatterBencher();
        void Start();
        int64_t Elapse();
        float Get();

    private:
#if (_MSC_VER)
        std::chrono::time_point<std::chrono::steady_clock> _start;
#else
        std::chrono::_V2::system_clock::time_point _start;
#endif // _WIN
        int64_t _elapse;
        bool _started = false;
        float _avg_fps;
        float _ratio;
    };

    namespace slog
    {

        /**
        * @class LogStreamEndLine
        * @brief The LogStreamEndLine class implements an end line marker for a log stream
        */
        class OV_MATTER_API LogStreamEndLine
        {
        };

        static constexpr LogStreamEndLine endl;

        /**
        * @class LogStreamBoolAlpha
        * @brief The LogStreamBoolAlpha class implements bool printing for a log stream
        */
        class OV_MATTER_API LogStreamBoolAlpha
        {
        };

        static constexpr LogStreamBoolAlpha boolalpha;

        /**
        * @class LogStream
        * @brief The LogStream class implements a stream for sample logging
        */
        class OV_MATTER_API LogStream
        {
            std::string _prefix;
            std::ostream *_log_stream;
            bool _new_line;

        public:
            /**
            * @brief A constructor. Creates a LogStream object
            * @param prefix The prefix to print
            */
            LogStream(const std::string &prefix, std::ostream &log_stream)
                : _prefix(prefix), _new_line(true)
            {
                _log_stream = &log_stream;
            }

            /**
            * @brief A stream output operator to be used within the logger
            * @param arg Object for serialization in the logger message
            */
            template <class T>
            LogStream &operator<<(const T &arg)
            {
                if (_new_line)
                {
                    (*_log_stream) << "[ " << _prefix << " ] ";
                    _new_line = false;
                }

                (*_log_stream) << arg;
                return *this;
            }

            // Specializing for LogStreamEndLine to support slog::endl
            LogStream &operator<<(const LogStreamEndLine & /*arg*/)
            {
                _new_line = true;

                (*_log_stream) << std::endl;
                return *this;
            }

            // Specializing for LogStreamBoolAlpha to support slog::boolalpha
            LogStream &operator<<(const LogStreamBoolAlpha & /*arg*/)
            {
                (*_log_stream) << std::boolalpha;
                return *this;
            }
        };

        static LogStream info("INFO", std::cout);
        static LogStream warn("WARNING", std::cout);
        static LogStream err("ERROR", std::cerr);
    } // namespace slog

    class OV_MATTER_API Utils_Ov
    {
    public:
        static std::string getMatType(const cv::Mat& img, bool more_info=true);
        static void showImage(cv::Mat& img, std::string& title);
        static void frameData2Mat(matter::FrameData& frameData, cv::Mat& outMat);
        static void mat2FrameData(cv::Mat& mat, matter::FrameData& frameData);

        static void sleep(long milliseconds);
        static double getSceneScore(cv::Mat prev_frame, cv::Mat frame, double& prev_mafd);

        static std::string getRealPath(std::string relPath);
    };
} // namespace ovlib

#endif