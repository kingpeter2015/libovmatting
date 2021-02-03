// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file with common samples functionality using OpenCV
 * @file ocv_common.hpp
 */

#pragma once

#include <common.hpp>
#include <opencv2/opencv.hpp>

/**
* @brief Sets image data stored in cv::Mat object to a given Blob object.
* @param orig_image - given cv::Mat object with an image data.
* @param blob - Blob object which to be filled by an image data.
* @param batchIndex - batch index of an image inside of the blob.
*/
template <typename T>
void matU8ToBlob(const cv::Mat &orig_image, InferenceEngine::Blob::Ptr &blob, int batchIndex = 0)
{
    InferenceEngine::SizeVector blobSize = blob->getTensorDesc().getDims();
    const size_t width = blobSize[3];
    const size_t height = blobSize[2];
    const size_t channels = blobSize[1];
    InferenceEngine::MemoryBlob::Ptr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
    if (!mblob)
    {
        THROW_IE_EXCEPTION << "We expect blob to be inherited from MemoryBlob in matU8ToBlob, "
                           << "but by fact we were not able to cast inputBlob to MemoryBlob";
    }
    // locked memory holder should be alive all time while access to its buffer happens
    auto mblobHolder = mblob->wmap();

    T *blob_data = mblobHolder.as<T *>();

    cv::Mat resized_image(orig_image);
    if (static_cast<int>(width) != orig_image.size().width ||
        static_cast<int>(height) != orig_image.size().height)
    {
        cv::resize(orig_image, resized_image, cv::Size(width, height));
    }

    int batchOffset = batchIndex * width * height * channels;

    for (size_t c = 0; c < channels; c++)
    {
        for (size_t h = 0; h < height; h++)
        {
            for (size_t w = 0; w < width; w++)
            {
                blob_data[batchOffset + c * width * height + h * width + w] =
                    resized_image.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
}

/**
 * @brief Wraps data stored inside of a passed cv::Mat object by new Blob pointer.
 * @note: No memory allocation is happened. The blob just points to already existing
 *        cv::Mat data.
 * @param mat - given cv::Mat object with an image data.
 * @return resulting Blob pointer.
 */
static UNUSED InferenceEngine::Blob::Ptr wrapMat2Blob(const cv::Mat &mat)
{
    size_t channels = mat.channels();
    size_t height = mat.size().height;
    size_t width = mat.size().width;

    size_t strideH = mat.step.buf[0];
    size_t strideW = mat.step.buf[1];

    bool is_dense = strideW == channels && strideH == channels * width;

    if (!is_dense)
    {
        THROW_IE_EXCEPTION << "Doesn't support conversion from not dense cv::Mat";
    }

    InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::U8, {1, channels, height, width}, InferenceEngine::Layout::NHWC);

    return InferenceEngine::make_shared_blob<uint8_t>(tDesc, mat.data);
}

static UNUSED InferenceEngine::Blob::Ptr wrapMat2Blob2(const cv::Mat &mat)
{
    cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
    cv::Mat fMat;
    mat.convertTo(fMat, CV_32FC3, 1.0/255.0);
    size_t channels = fMat.channels();
    size_t channelsize = channels * sizeof(float);
    size_t height = fMat.size().height;
    size_t width = fMat.size().width;

    size_t strideH = fMat.step.buf[0];
    size_t strideW = fMat.step.buf[1];
    
    size_t number_of_data = height * width * channels * sizeof(float); 

    bool is_dense = strideW == channelsize && strideH == channelsize * width;

    if (!is_dense)
    {
        THROW_IE_EXCEPTION << "Doesn't support conversion from not dense cv::Mat";
    }

    InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::FP32, {1, channels, height, width}, InferenceEngine::Layout::NHWC);

    return InferenceEngine::make_shared_blob<float>(tDesc, (float*)fMat.data, number_of_data);
}
