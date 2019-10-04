#ifndef RETINAFACE_H__
#define RETINAFACE_H__


#include "tensorNet.h"
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
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

    void print() {
        // printf("rect %f %f %f %f\n", val[0], val[1], val[2], val[3]);
    }
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
        assert(0 <= i && i <= 4);

        if (i == 0) 
            return finalbox.x;
        if (i == 1) 
            return finalbox.y;
        if (i == 2) 
            return finalbox.width;
        if (i == 3) 
            return finalbox.height;
    }

    float operator[](int i) const {
        assert(0 <= i && i <= 4);

        if (i == 0) 
            return finalbox.x;
        if (i == 1) 
            return finalbox.y;
        if (i == 2) 
            return finalbox.width;
        if (i == 3) 
            return finalbox.height;
    }

    cv::Rect_< float > anchor; // x1,y1,x2,y2
	float reg[4]; // offset reg
    cv::Point center; // anchor feat center
	float score; // cls score
    std::vector<cv::Point2f> pts; // pred pts

    cv::Rect_< float > finalbox; // final box res

    void print() {
        // printf("finalbox %f %f %f %f, score %f\n", finalbox.x, finalbox.y, finalbox.width, finalbox.height, score);
        // printf("landmarks ");
        // for (int i = 0; i < pts.size(); ++i) {
            // printf("%f %f, ", pts[i].x, pts[i].y);
        // }
        // printf("\n");
    }
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

    void bbox_pred(const CRect2f& anchor, const CRect2f& delta, cv::Rect_< float >& box);

    void landmark_pred(const CRect2f anchor, const std::vector<cv::Point2f>& delta, std::vector<cv::Point2f>& pts);

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

