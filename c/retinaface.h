#ifndef RETINAFACE_H__
#define RETINAFACE_H__


#include "tensorNet.h"
#include <memory>
#include <vector>
#include <map>
#include <algorithm>
#include <glog/logging.h>

using std::vector;

/**
 * Object recognition and localization networks with TensorRT support.
 * @ingroup detectNet
 */

class AnchorCfg {
public:
	  std::vector<float> SCALES;	
	  std::vector<float> RATIOS;
	  int BASE_SIZE;

    AnchorCfg() {}
    ~AnchorCfg() {}
	  AnchorCfg(const std::vector<float> s, const std::vector<float> r, int size) {
      SCALES = s;
      RATIOS = r;
      BASE_SIZE = size;
	  }
};

class CRect2f {
public:
    CRect2f(float x1, float y1, float x2, float y2) {
        val[0] = x1;
        val[1] = y1;
        val[2] = x2;
        val[3] = y2;
    }

    float& operator[](int i) {
        return val[i];
    }

    float operator[](int i) const {
        return val[i];
    }
    float val[4];
};

class CPointf {
public:
    CPointf(float x, float y): x(x), y(y) {
    }
    CPointf() = default;

    float x;
    float y;
};

class Anchor {
public:
	Anchor() {
	}

	~Anchor() {
	}

  bool operator<(const Anchor &t) const {
      return score < t.score;
  }

  bool operator>(const Anchor &t) const {
      return score > t.score;
  }

  float& operator[](int i) {
      return finalbox[i];

  }

  float operator[](int i) const {
      return finalbox[i];
  }

  float anchor[4]; // x1,y1,x2,y2
	float reg[4]; // offset reg
  // cv::Point center; // anchor feat center
	float score; // cls score
  std::vector<CPointf> pts; // pred pts
  // final box res
  float finalbox[4];
};

class AnchorGenerator {
public:
	AnchorGenerator();
	~AnchorGenerator();

    // init different anchors
    int Init(int stride, const AnchorCfg& cfg, bool dense_anchor);

    // anchor plane
    // int Generate(int fwidth, int fheight, int stride, float step, std::vector<int>& size, std::vector<float>& ratio, bool dense_anchor);

	// filter anchors and return valid anchors
    int FilterAnchor(const float* cls_data,  const float* reg_data, const float* pts_data, const int h, const int w, std::vector<Anchor>& result);


private:
    void _ratio_enum(const CRect2f& anchor, const std::vector<float>& ratios, std::vector<CRect2f>& ratio_anchors);

    void _scale_enum(const std::vector<CRect2f>& ratio_anchor, const std::vector<float>& scales, std::vector<CRect2f>& scale_anchors);

    void bbox_pred(const CRect2f& anchor, const CRect2f& delta, float* box);

    void landmark_pred(const CRect2f anchor, const std::vector<CPointf>& delta, std::vector<CPointf>& pts);

	std::vector<std::vector<Anchor>> anchor_planes; // corrspont to channels

	std::vector<int> anchor_size; 
	std::vector<float> anchor_ratio;
	float anchor_step; // scale step
	int anchor_stride; // anchor tile stride
	int feature_w; // feature map width
	int feature_h; // feature map height

    std::vector<CRect2f> preset_anchors;
	int anchor_num; // anchor type num

    const float cls_threshold = 0.8;
    // const float nms_threshold = 0.4;
};

class retinaface : public tensorNet
{
public:
    retinaface();
    ~retinaface();

	int Detect( float* input, uint32_t width, uint32_t height, std::vector<Anchor>& detections);

	static std::shared_ptr<retinaface> Create( const char* prototxt_path, const char* model_path, 
						 uint32_t maxBatchSize=1, 
						 precisionType precision=TYPE_FASTEST,
				   		 deviceType device=DEVICE_GPU, bool allowGPUFallback=true, nvinfer1::IInt8Calibrator* calibrator=NULL);
private:
    const float mMeans[3] = {0, 0, 0};
    const char *m_input_name = "data";
    std::vector<int> _feat_stride_fpn = {32, 16, 8};
    std::map<int, AnchorCfg> anchor_cfg = {
        {32, AnchorCfg(std::vector<float>{32,16}, std::vector<float>{1}, 16)},
        {16, AnchorCfg(std::vector<float>{8,4}, std::vector<float>{1}, 16)},
        {8,AnchorCfg(std::vector<float>{2,1}, std::vector<float>{1}, 16)}
    };
    vector<std::string> m_output_names;
    float cls_threshold = 0.8;
    float nms_threshold = 0.4;
    std::vector<AnchorGenerator> ac_;
    void* inferenceBuffers[10];
    int anchor_num_;
};

void nms_cpu(std::vector<Anchor>& boxes, float threshold, std::vector<Anchor>& filterOutBoxes);

#endif

