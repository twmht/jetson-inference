/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "retinaface.h"
#include "EntropyCalibrator.h"
#include "loadImage.h"

#include "commandLine.h"
#include "cudaMappedMemory.h"
#include <glog/logging.h>
#include <dirent.h>
#include <sys/stat.h>

bool fileExist(const std::string& file) {
    struct stat buffer;
    return stat(file.c_str(), &buffer) == 0;
}

void readImages(std::vector<std::string>& images, const std::string& filePath, int* usedImageNum) {
    int count = 0;
    DIR* root = opendir(filePath.c_str());
    if (root == NULL) {
        LOG(ERROR) << ("open %s failed!\n", filePath.c_str());
        return;
    }
    struct dirent* ent = readdir(root);
    while (ent != NULL) {
        if (ent->d_name[0] != '.') {
            const std::string fileName = filePath + "/" + ent->d_name;
            if (fileExist(fileName)) {
                // std::cout << "==> " << fileName << std::endl;
                // DLOG(INFO) << fileName;
                if (*usedImageNum == 0) {
                    // use all images in the folder
                    images.push_back(fileName);
                    count++;
                } else if (count < *usedImageNum) {
                    // use usedImageNum images
                    images.push_back(fileName);
                    count++;
                } else {
                    break;
                }
            }
        }
        ent = readdir(root);
    }
    if (*usedImageNum == 0) {
        *usedImageNum = count;
    }
    LOG(INFO) << "used image num: " << images.size();
}

int main( int argc, char** argv )
{

	/*
	 * create detection network
	 */


  std::vector<std::string> images;
  int usedImageNum = 0;
  const char* networkName = "Retinaface";
  std::string filePath = "/home/acer/wider_train_images";
  readImages(images, filePath,  &usedImageNum);

  const int batchSize = 128;
  const int batchRuns = images.size() / batchSize;

  const int imgWidth = 300;
  const int imgHeight = 300;
  std::string table_name = "/home/acer/trt_models/CalibrationTableRetinaface";

  std::unique_ptr<nvinfer1::IInt8Calibrator> calibrator;
  calibrator.reset(new Int8EntropyCalibrator2(table_name));

  std::shared_ptr<retinaface> net = retinaface::Create("/home/acer/trt_models/mnet_rt.prototxt", "/home/acer/trt_models/mnet.caffemodel", batchSize, TYPE_FASTEST, DEVICE_GPU, true, calibrator.get());
  calibrator.release();

	return 0;
}


