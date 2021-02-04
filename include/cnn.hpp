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
#include <ie_common.h>

/**
* @brief Base class of config for network
*/
struct CNNConfig
{
    explicit CNNConfig(const std::string &path_to_model, const std::string &bin, const cv::Size shape=cv::Size(256,144)) : path_to_model(path_to_model), path_to_bin(bin), input_shape(shape){}

    /** @brief Path to model description */
    std::string path_to_model;
    std::string path_to_bin;
    float scale{0.25};
    /** @brief Maximal size of batch */
    int max_batch_size{1};

    /** @brief Inference Engine */
    InferenceEngine::Core ie;
    /** @brief Device name */
    std::string deviceName{"CPU"};
    
    bool is_async{true};
    cv::Size input_shape;
    int cpu_threads_num;    //default 0
    bool cpu_bind_thread; //default true
    int cpu_throughput_streams;    //default 1
};

/**
* @brief Base class of network
*/
class CnnDLSDKBase
{
public:
    /**
   * @brief Constructor
   */
    explicit CnnDLSDKBase(const CNNConfig& config);

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
    CNNConfig _config;

    /** @brief Net inputs info */
    InferenceEngine::InputsDataMap _inInfo;
    /** @brief Name of the input blob input blob */
    std::vector<std::string> _input_blob_names;
    
    /** @brief Net outputs info */
    InferenceEngine::OutputsDataMap _outInfo;
    /** @brief Names of output blobs */
    std::vector<std::string> _output_blobs_names;
    
    InferenceEngine::CNNNetwork _cnn_network_;
    
    /** @brief IE network */
    InferenceEngine::ExecutableNetwork _executable_network_;
    /** @brief IE InferRequest */
    mutable InferenceEngine::InferRequest _infer_request_;
    std::map<std::string, InferenceEngine::SizeVector> _input_shapes;
    
    
};

class MattingCNN : public CnnDLSDKBase
{
public:
    explicit MattingCNN(const CNNConfig& config);

    void Compute(const cv::Mat &image, cv::Mat &bgr, std::map<std::string, cv::Mat> *result, cv::Size& outp_shape) const;
    void Compute2(const cv::Mat &image, cv::Mat &bgr, cv::Mat &bgr2, std::map<std::string, cv::Mat> *result, cv::Size& outp_shape) const;

    void Compute_Alpha(const cv::Mat &image, cv::Mat &bgr,  std::map<std::string, cv::Mat> *result, cv::Size& outp_shape) const;

private:
    cv::Size _originShape;
};

//异步算法
template <typename T>
class AsyncAlgorithm
{
public:
    virtual ~AsyncAlgorithm() {}
    virtual void enqueue(const cv::Mat& frame, const cv::Mat& bgr, const cv::Mat& bgrReplace, const cv::Size& out_shape) = 0;
    virtual void submitRequest() = 0;
    virtual void wait() = 0;
    virtual void printPerformanceCounts(const std::string &fullDeviceName) = 0;
    virtual std::vector<T> fetchResults() = 0;
};

template <typename T>
class BaseAsyncCNN : public AsyncAlgorithm<T>
{
protected:
    InferenceEngine::InferRequest::Ptr request;
    bool isAsync;
    std::string topoName;

public:
    BaseAsyncCNN(){}

    void submitRequest() override
    {
        if (request == nullptr)
        {
            return;
        }

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
        {
            return;
        }
        request->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
    }

    void printPerformanceCounts(const std::string &fullDeviceName) override
    {
        std::cout << "BaseMatting Performance counts for " << topoName << std::endl << std::endl;
        ::printPerformanceCounts(*request, std::cout, fullDeviceName, false);
    }
};
