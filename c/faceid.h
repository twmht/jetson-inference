#ifndef __FACEID_H__
#define __FACEID_H__


#include "tensorNet.h"
#include <memory>
#include <vector>

using std::vector;


/**
 * Object recognition and localization networks with TensorRT support.
 * @ingroup detectNet
 */
class faceid : public tensorNet
{
public:
    faceid();
    ~faceid();

	// bool init( const char* prototxt_path, const char* model_path, float threshold, const char* input, const char* bboxes, uint32_t maxBatchSize, 
			 // precisionType precision, deviceType device, bool allowGPUFallback);

	static std::shared_ptr<faceid> Create( const char* prototxt_path, const char* model_path, 
						 uint32_t maxBatchSize=1, 
						 precisionType precision=TYPE_FASTEST,
				   		 deviceType device=DEVICE_GPU, bool allowGPUFallback=true);
    int Detect( float* input, uint32_t width, uint32_t height, float**);
private:
    const float mMeans[3] = {127.5, 127.5, 127.5};
    const float mStds[3] = {0.0078125f, 0.0078125f, 0.0078125f};
    const char *m_input_name = "data";
    const int mWidth = 112;
    const int mHeight = 112;
	void* inferenceBuffers[2];
    vector<std::string> m_output_names = {"fc"};
};


#endif


