#include "faceid.h"
#include "imageNet.cuh"

faceid::faceid() : tensorNet()
{
}

faceid::~faceid()
{
}

std::shared_ptr<faceid> faceid::Create( const char* prototxt_path, const char* model_path, 
                 uint32_t maxBatchSize,
                 precisionType precision,
                 deviceType device, bool allowGPUFallback) {
    std::shared_ptr<faceid> net = std::make_shared<faceid>();
    // net->init(prototxt_path, model_path)
    net->LoadNetwork(prototxt_path, model_path, NULL, net->m_input_name, net->m_output_names, maxBatchSize, precision, device, allowGPUFallback);
    return net;
}

int faceid::Detect( float* input, uint32_t width, uint32_t height, float** emb) {

    if( CUDA_FAILED(cudaPreImageNetNormBGR((float4*)input, width, height, mInputCUDA, mWidth, mHeight,
                                      make_float2(0, 255), GetStream())) )
    {
        printf(LOG_TRT "faceid::Detect() -- cudaPreImageNetNorm() failed\n");
        return -1;
    }
    inferenceBuffers[0] = mInputCUDA;
    for (int i = 0; i < mOutputs.size() ; i++) {
        const int outputIndex = mEngine->getBindingIndex(mOutputs[i].name.c_str());
        inferenceBuffers[outputIndex] = mOutputs[i].CUDA;
    }
    // PROFILER_BEGIN(PROFILER_NETWORK);
    if( !mContext->execute(1, inferenceBuffers) )
    {
        printf(LOG_TRT "faceid::Detect() -- failed to execute TensorRT context\n");
        return -1;
    }

    *emb = mOutputs[0].CUDA;
    return 0;
}
