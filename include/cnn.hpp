#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include <time.h>

#include "ocv_common.hpp"

#include <inference_engine.hpp>


class TimerCounter
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
        _elapse = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
        std::cout << "Benchmarking " << _name <<"\t\t| elapse miliseconds: " << _elapse << ",\t estimate fps: " << 1000.0/_elapse << std::endl;
    }
private:
    std::string _name;
    std::chrono::_V2::system_clock::time_point _start;
    int64_t _elapse;
    bool _started = false;
};

class FaceTimerCounter
{
public:
  FaceTimerCounter();
  virtual ~FaceTimerCounter();
  void Start();
  int64_t Elapse();

private:
  std::chrono::_V2::system_clock::time_point _start;
  int64_t _elapse;
  bool _started = false;
};

struct CNetWorkCPUConfig
{
    int nCpuThreadsNum;        //default 0
    bool bCpuBindThread;       //default true
    int nCpuThroughputStreams; //default 1
};

/**
* @brief Base class of config for network
*/
struct CnnConfig
{
    explicit CnnConfig(const std::string &path_to_model, const std::string &bin, const std::string &src, const std::string &bgr, const cv::Size shape=cv::Size(1920,1080)) : path_to_model(path_to_model), path_to_bin(bin), path_bgr(bgr), path_src(src), _shape(shape){}

    /** @brief Path to model description */
    std::string path_to_model;
    std::string path_to_bin;
    //背景图片
    std::string path_bgr;
    //视频路径
    std::string path_src;
    float scale{0.25};
    /** @brief Maximal size of batch */
    int max_batch_size{1};

    /** @brief Inference Engine */
    InferenceEngine::Core ie;
    /** @brief Device name */
    std::string deviceName{"CPU"};
    CNetWorkCPUConfig networkCfg;
    
    bool is_async{true};
    cv::Size _shape = cv::Size(1920, 1080);
};

/**
* @brief Base class of network
*/
class CnnDLSDKBase
{
public:
    using Config = CnnConfig;

    /**
   * @brief Constructor
   */
    explicit CnnDLSDKBase(const Config &config);

    /**
   * @brief Descructor
   */
    ~CnnDLSDKBase() {}

    /**
   * @brief Loads network
   */
    void Load();

    /**
    * @brief Prints performance report
    */
    void PrintPerformanceCounts(std::string fullDeviceName) const;

protected:
    
    /** @brief Config */
    Config _config;

    /** @brief Net inputs info */
    InferenceEngine::InputsDataMap _inInfo;
    /** @brief Name of the input blob input blob */
    std::vector<std::string> _input_blob_names;
    
    /** @brief Net outputs info */
    InferenceEngine::OutputsDataMap _outInfo;
    /** @brief Names of output blobs */
    std::vector<std::string> _output_blobs_names;
    
    /** @brief IE network */
    InferenceEngine::ExecutableNetwork _executable_network_;
    /** @brief IE InferRequest */
    mutable InferenceEngine::InferRequest _infer_request_;
    std::map<std::string, InferenceEngine::SizeVector> _input_shapes;
    
    
};

class MattingCNN : public CnnDLSDKBase
{
public:
    explicit MattingCNN(const CnnConfig &config);

    void Compute(const cv::Mat &image, cv::Mat &bgr,  std::map<std::string, cv::Mat> *result, cv::Size& outp_shape) const;
};

//异步算法
class AsyncAlgorithm
{
public:
    virtual ~AsyncAlgorithm() {}
    virtual void enqueue(const cv::Mat &frame, const cv::Mat& image) = 0;
    virtual void submitRequest() = 0;
    virtual void wait() = 0;
    virtual void printPerformanceCounts(const std::string &fullDeviceName) = 0;
};

//
template <typename T>
class AsyncDetection : public AsyncAlgorithm
{
public:
    virtual std::vector<T> fetchResults() = 0;
};

class BaseMatting : public AsyncAlgorithm
{
protected:
    InferenceEngine::InferRequest::Ptr request;
    const bool isAsync;
    std::string topoName;

public:
    explicit BaseMatting(bool isAsync = false) : isAsync(isAsync) {}

    void submitRequest() override
    {
        if (request == nullptr)
            return;
        if (isAsync)
        {
            request->StartAsync();
        }
        else
        {
            request->Infer();
        }
    }

    void wait() override
    {
        if (!request || !isAsync)
            return;
        request->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
    }

    void printPerformanceCounts(const std::string &fullDeviceName) override
    {
        std::cout << "BaseMatting Performance counts for " << topoName << std::endl
                  << std::endl;
        ::printPerformanceCounts(*request, std::cout, fullDeviceName, false);
    }
};
