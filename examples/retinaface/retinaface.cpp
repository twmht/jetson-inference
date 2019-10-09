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
#include "loadImage.h"

#include "commandLine.h"
#include "cudaMappedMemory.h"
#include "EntropyCalibrator.h"
#include <opencv2/opencv.hpp>


int main( int argc, char** argv )
{

	/*
	 * create detection network
	 */

  std::string table_name = "/home/acer/trt_models/CalibrationTableRetinaface";

  std::unique_ptr<nvinfer1::IInt8Calibrator> calibrator;
  calibrator.reset(new Int8EntropyCalibrator2(table_name));

    // std::shared_ptr<retinaface> net = retinaface::Create("/home/alec.tu/RetinaFace-Cpp/convert_models/mnet/mnet_rt.prototxt", "/home/alec.tu/RetinaFace-Cpp/convert_models/mnet/mnet.caffemodel");
  const int batchSize = 1;
  std::shared_ptr<retinaface> net = retinaface::Create("/home/acer/trt_models/mnet_rt.prototxt", "/home/acer/trt_models/mnet.caffemodel", batchSize, TYPE_FASTEST, DEVICE_GPU, true, calibrator.get());
  calibrator.release();

	if( !net )
	{
		printf("detectnet-console:   failed to initialize detectNet\n");
		return 0;
	}

    net->EnableLayerProfiler();
	
	
	/*
	 * load image from disk
	 */
    float* imgCPU    = NULL;
    float* imgCUDA   = NULL;
    int    imgWidth  = 300;
    int    imgHeight = 300;
    const char * imgFilename = "/home/acer/mnn_demo/app/models/3.jpg";

    cv::Mat ori_img = cv::imread(imgFilename);
    float w_scale = (float)ori_img.cols / imgWidth;
    float h_scale = (float)ori_img.rows / imgHeight;

		
    if( !loadImageRGBA(imgFilename, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight) )
    {
        printf("failed to load image '%s'\n", imgFilename);
        return 0;
    }

    // if( !saveImageRGBA("debug.jpg", (float4*)imgCPU, imgWidth, imgHeight, 255.0f) )
        // printf("error save\n");
    // }
	// saveImageRGBA("debug.jpg", (float4*)imgCPU, imgWidth, imgHeight, 255.0f);

	/*
	 * detect objects in image
	 */
	// detectNet::Detection* detections = NULL;

    std::vector<Anchor> finalBbox;
    const int numDetections = net->Detect(imgCUDA, imgWidth, imgHeight, finalBbox);
    LOG(INFO) << "result: " << finalBbox.size();

    for(int i = 0; i < finalBbox.size(); i ++)
    {
        finalBbox[i].finalbox[0] *= w_scale;
        finalBbox[i].finalbox[1] *= h_scale;
        finalBbox[i].finalbox[2] *= w_scale;
        finalBbox[i].finalbox[3] *= h_scale;

        for (int j = 0; j < finalBbox[i].pts.size(); ++j) {
            finalBbox[i].pts[j].x *= w_scale;
            finalBbox[i].pts[j].y *= h_scale;
        }
    }

    for(int i = 0; i < finalBbox.size(); i ++)
    {
        cv::rectangle (ori_img, cv::Point((int)finalBbox[i].finalbox[0], (int)finalBbox[i].finalbox[1]), cv::Point((int)finalBbox[i].finalbox[2], (int)finalBbox[i].finalbox[3]), cv::Scalar(0, 255, 255), 2, 8, 0);
        for (int j = 0; j < finalBbox[i].pts.size(); ++j) {
            cv::circle(ori_img, cv::Point((int)finalBbox[i].pts[j].x, (int)finalBbox[i].pts[j].y), 1, cv::Scalar(225, 0, 225), 2, 8);
        }
    }
    cv::imwrite("rt_result.jpg", ori_img);

	// // print out the detection results
	// printf("%i objects detected\n", numDetections);
	
	// for( int n=0; n < numDetections; n++ )
	// {
		// printf("detected obj %u  class #%u (%s)  confidence=%f\n", detections[n].Instance, detections[n].ClassID, net->GetClassDesc(detections[n].ClassID), detections[n].Confidence);
		// printf("bounding box %u  (%f, %f)  (%f, %f)  w=%f  h=%f\n", detections[n].Instance, detections[n].Left, detections[n].Top, detections[n].Right, detections[n].Bottom, detections[n].Width(), detections[n].Height()); 
	// }
	
	// // wait for the GPU to finish		
	// CUDA(cudaDeviceSynchronize());

	// // print out timing info
	// net->PrintProfilerTimes();
	
	// // save image to disk
	// const char* outputFilename = cmdLine.GetPosition(1);
	
	// if( outputFilename != NULL )
	// {
		// printf("detectnet-console:  writing %ix%i image to '%s'\n", imgWidth, imgHeight, outputFilename);
		
		// if( !saveImageRGBA(outputFilename, (float4*)imgCPU, imgWidth, imgHeight, 255.0f) )
			// printf("detectnet-console:  failed saving %ix%i image to '%s'\n", imgWidth, imgHeight, outputFilename);
		// else	
			// printf("detectnet-console:  successfully wrote %ix%i image to '%s'\n", imgWidth, imgHeight, outputFilename);
	// }


	/*
	 * destroy resources
	 */
	// printf("detectnet-console:  shutting down...\n");

	// CUDA(cudaFreeHost(imgCPU));
	// SAFE_DELETE(net);

	// printf("detectnet-console:  shutdown complete\n");
	return 0;
}

