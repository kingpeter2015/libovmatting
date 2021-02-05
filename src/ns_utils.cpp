#include "ns_utils.hpp"
#include <cassert>

using namespace ovlib;

MatterBencher::MatterBencher() 
{
    _avg_fps = -1.0f;
    _ratio = 0.5f;
}

MatterBencher::~MatterBencher()
{
}

void MatterBencher::Start()
{
    if (_started)
    {
        return;
    }
    _elapse = 0;
    _start = std::chrono::high_resolution_clock::now();
    _started = true;
}

int64_t MatterBencher::Elapse()
{
    auto ckpnt = std::chrono::high_resolution_clock::now();
    auto elapsed = ckpnt - _start;
    _elapse = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    _start = ckpnt;

    float fps_sample = 1000.0 / (_elapse);
    _avg_fps = _avg_fps < 0.0f ? fps_sample : (_ratio * fps_sample + (1 - _ratio) * _avg_fps);

    return _elapse;
}

float MatterBencher::Get()
{
    return _avg_fps;
}

std::vector<std::string> Utils_String::split(const std::string &s, char delim)
{
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;

    while (getline(ss, item, delim))
    {
        result.push_back(item);
    }
    return result;
}

std::string Utils_Ov::getMatType(const cv::Mat &img, bool more_info)
{
    std::string r;
    int type = img.type();
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth)
    {
    case CV_8U:
        r = "8U";
        break;
    case CV_8S:
        r = "8S";
        break;
    case CV_16U:
        r = "16U";
        break;
    case CV_16S:
        r = "16S";
        break;
    case CV_32S:
        r = "32S";
        break;
    case CV_32F:
        r = "32F";
        break;
    case CV_64F:
        r = "64F";
        break;
    default:
        r = "User";
        break;
    }

    r += "C";
    r += (chans + '0');

    if (more_info)
    {
        std::cout << "depth: " << img.depth() << " channels: " << img.channels() << std::endl;
    }

    return r;
}

void Utils_Ov::showImage(cv::Mat& img, std::string& title)
{
    std::string image_type = getMatType(img);
    cv::namedWindow(title + " type:" + image_type, cv::WINDOW_NORMAL); // Create a window for display.
    cv::imshow(title, img);
    cv::waitKey(0);
}


void Utils_Ov::mat2FrameData(cv::Mat& mat, matter::FrameData& frameData)
{
    if (mat.empty())
    {
        frameData.width = 0;
        frameData.height = 0;
        frameData.frame = 0;
        return;
    }
    cv::Mat frame = mat.clone();
    frameData.width = mat.cols;
    frameData.height = mat.rows;
    frameData.frame = frame.data;
    int type = mat.type();
    if (type == CV_8UC3)
    {
        frameData.format = matter::FRAME_FOMAT_BGR;
        size_t size = 3 * frameData.width * frameData.height;
        frameData.frame = new unsigned char[size];
        memcpy(frameData.frame, mat.data, size);
    }
    else if (type == CV_8UC1)
    {
        frameData.format = matter::FRAME_FOMAT_GRAY;
        size_t size = frameData.width * frameData.height;
        frameData.frame = new unsigned char[size];
        memcpy(frameData.frame, mat.data, size);
    }
}


void Utils_Ov::frameData2Mat(matter::FrameData& frameData, cv::Mat& outMat)
{
    if (frameData.format == 0 || frameData.width <= 0 || frameData.height <= 0)
    {
        outMat = cv::Mat();
    }

    if (frameData.format == matter::FRAME_FOMAT_I420)
    {
        size_t size = frameData.height * frameData.width * 3 / 2;
        outMat = cv::Mat(frameData.height + frameData.height / 2, frameData.width, CV_8UC1, frameData.frame);
        cv::cvtColor(outMat, outMat, cv::COLOR_YUV2BGR_I420);
    }
    else if (frameData.format == matter::FRAME_FOMAT_RGB)
    {
        size_t size = frameData.height * frameData.width * 3;
        outMat = cv::Mat(frameData.height, frameData.width, CV_8UC3, frameData.frame);
        cv::cvtColor(outMat, outMat, cv::COLOR_RGB2BGR);
    }
    else if (frameData.format == matter::FRAME_FOMAT_BGR)
    {
        size_t size = frameData.height * frameData.width * 3;
        outMat = cv::Mat(frameData.height, frameData.width, CV_8UC3, frameData.frame);
    }
    else if (frameData.format == matter::FRAME_FOMAT_GRAY)
    {
        size_t size = frameData.height * frameData.width;
        outMat = cv::Mat(frameData.height, frameData.width, CV_8UC1, frameData.frame);
    }
}

void Utils_Ov::sleep(long milliseconds)
{
    std::chrono::milliseconds dura(milliseconds);
    std::this_thread::sleep_for(dura);
}