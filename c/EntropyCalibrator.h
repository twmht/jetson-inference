/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ENTROPY_CALIBRATOR_H
#define ENTROPY_CALIBRATOR_H

#include <NvInfer.h>
#include <iostream>
#include <fstream>
#include <iterator>
#include <glog/logging.h>
#include "cudaMappedMemory.h"
#include "loadImage.h"

//! \class EntropyCalibratorImpl
//!
//! \brief Implements common functionality for Entropy calibrators.
//!
class EntropyCalibratorImpl
{
public:
    EntropyCalibratorImpl( int batchSize, int batchRuns, int height, int width, std::string table, std::vector<std::string>& images, bool cache_read)
        : mCalibrationTableName(table)
        , mBatchSize(batchSize)
        , mBatchRuns(batchRuns)
        , mHeight(height)
        , mWidth(width)
        , mImages(images)
        , mCacheRead(cache_read)
    {
      size_t inputSize = mBatchSize * 3 * mHeight * mWidth * sizeof(float);
      CHECK(cudaAllocMapped((void**)&mInputCPU, (void**)&mInputCUDA, inputSize));
    }

    EntropyCalibratorImpl(std::string table)
        : mCalibrationTableName(table)
        , mCacheRead(true)
    {
    }

    virtual ~EntropyCalibratorImpl()
    {
      cudaFreeHost(mInputCPU);
    }

    int getBatchSize() const
    {
        return mBatchSize;
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings)
    {
      if (mCurrentCount == mBatchRuns) {
        return false;
      }
      const int dim = mWidth * mHeight * 3;
      for (int i = 0; i < mBatchSize; i++) {
        const char* imgFilename = mImages[mCurrentCount * mBatchSize + i].c_str();
        if( !loadImageRGB_(imgFilename, mInputCPU + i * dim, &mWidth, &mHeight) )
        {
          LOG(ERROR) << "failed to load image " << imgFilename;
          return 0;
        }
      }
      LOG(INFO) << "Calibration #" << mCurrentCount << "\t" << mCurrentCount << "/" << mBatchRuns;
      bindings[0] = mInputCUDA;
      mCurrentCount++;
      return true;
    }

    const void* readCalibrationCache(size_t& length)
    {
      mCalibrationCache.clear();
      std::ifstream input(mCalibrationTableName, std::ios::binary);
      input >> std::noskipws;
      if (mCacheRead && input.good())
      {
          std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
              std::back_inserter(mCalibrationCache));
      }
      length = mCalibrationCache.size();
      return length ? mCalibrationCache.data() : nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length)
    {
        std::ofstream output(mCalibrationTableName, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }

private:
    std::string mCalibrationTableName;
    std::vector<char> mCalibrationCache;
    std::vector<std::string> mImages;
    int mBatchSize = 0;
    int mBatchRuns = 0;
    int mHeight, mWidth;
    int mCurrentCount = 0;
    float *mInputCUDA, *mInputCPU;
    bool mCacheRead = false;
};

class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    Int8EntropyCalibrator2( int batchSize, int batchRuns, int height, int width, std::string table, std::vector<std::string>& images, bool cache_read)
        : mImpl(batchSize, batchRuns, height, width, table, images, cache_read)
    {
    }

    Int8EntropyCalibrator2(std::string table)
        : mImpl(table)
    {
    }

    int getBatchSize() const override
    {
        return mImpl.getBatchSize();
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override
    {
        return mImpl.getBatch(bindings, names, nbBindings);
    }

    const void* readCalibrationCache(size_t& length) override
    {
        return mImpl.readCalibrationCache(length);
    }

    void writeCalibrationCache(const void* cache, size_t length) override
    {
        mImpl.writeCalibrationCache(cache, length);
    }

private:
    EntropyCalibratorImpl mImpl;
};

#endif // ENTROPY_CALIBRATOR_H
