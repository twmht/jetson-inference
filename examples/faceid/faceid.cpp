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

#include "faceid.h"
#include "loadImage.h"

#include "commandLine.h"
#include "cudaMappedMemory.h"


int main( int argc, char** argv )
{

	/*
	 * create detection network
	 */

    std::shared_ptr<faceid> net = faceid::Create("/home/alec.tu/facenet-pytorch/20190915_fmobilenet.prototxt", "/home/alec.tu/facenet-pytorch/20190915_fmobilenet.caffemodel");

	if( !net )
	{
		printf("detectnet-console:   failed to initialize detectNet\n");
		return 0;
	}

	//net->EnableLayerProfiler();
	
	
	/*
	 * load image from disk
	 */
    float* imgCPU    = NULL;
    float* imgCUDA   = NULL;
    int    imgWidth  = 112;
    int    imgHeight = 112;
    const char * imgFilename = "/home/alec.tu/gooface_quantize/0.jpg";
		
    if( !loadImageRGBA(imgFilename, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight) )
    {
        printf("failed to load image '%s'\n", imgFilename);
        return 0;
    }
	

	/*
	 * detect objects in image
	 */
    float* emb;
    net->Detect(imgCUDA, imgWidth, imgHeight, &emb);

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


