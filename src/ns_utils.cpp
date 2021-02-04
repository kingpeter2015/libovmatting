#include "ns_utils.hpp"
#include <cassert>

using namespace ovlib;

FaceTimerCounter::FaceTimerCounter() {}

FaceTimerCounter::~FaceTimerCounter()
{
}

void FaceTimerCounter::Start()
{
    if (_started)
    {
        return;
    }
    _elapse = 0;
    _start = std::chrono::high_resolution_clock::now();
    _started = true;
}

int64_t FaceTimerCounter::Elapse()
{
    auto ckpnt = std::chrono::high_resolution_clock::now();
    auto elapsed = ckpnt - _start;
    _elapse = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    _start = ckpnt;

    return _elapse;
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
    //matter::FrameData frameData;
    frameData.width = mat.cols;
    frameData.height = mat.rows;
    frameData.frame = mat.data;
    int type = mat.type();
    if (type == CV_8UC3)
    {
        frameData.format = matter::FRAME_FOMAT_BGR;
    }
    else if (type == CV_8UC1)
    {
        frameData.format = matter::FRAME_FOMAT_GRAY;
    }
}


void Utils_Ov::frameData2Mat(matter::FrameData& frameData, cv::Mat& outMat)
{
    assert(frameData.frame != 0);
    assert(frameData.width != 0);
    assert(frameData.height != 0);

    if (frameData.format == matter::FRAME_FOMAT_I420)
    {
        size_t size = frameData.height * frameData.width * 3 / 2;
        outMat = cv::Mat(frameData.height + frameData.height / 2, frameData.width, CV_8UC1);
        memcpy(outMat.data, frameData.frame, size);
        cv::cvtColor(outMat, outMat, cv::COLOR_YUV2BGR_I420);
    }
    else if (frameData.format == matter::FRAME_FOMAT_RGB)
    {
        size_t size = frameData.height * frameData.width * 3;
        outMat = cv::Mat(frameData.height, frameData.width, CV_8UC3);
        memcpy(outMat.data, frameData.frame, size);
        cv::cvtColor(outMat, outMat, cv::COLOR_RGB2BGR);
    }
    else if (frameData.format == matter::FRAME_FOMAT_BGR)
    {
        size_t size = frameData.height * frameData.width * 3;
        outMat = cv::Mat(frameData.height, frameData.width, CV_8UC3);
           
        memcpy(outMat.data, frameData.frame, size);
    }
    else if (frameData.format == matter::FRAME_FOMAT_GRAY)
    {
        size_t size = frameData.height * frameData.width;
        outMat = cv::Mat(frameData.height, frameData.width, CV_8UC1);

        memcpy(outMat.data, frameData.frame, size);
    }
}