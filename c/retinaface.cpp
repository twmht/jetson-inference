#include "retinaface.h"
#include "imageNet.cuh"
#include "loadImage.h"

void nms_cpu(std::vector<Anchor>& boxes, float threshold, std::vector<Anchor>& filterOutBoxes) {
    filterOutBoxes.clear();
    if(boxes.size() == 0)
        return;
    std::vector<size_t> idx(boxes.size());

    for(unsigned i = 0; i < idx.size(); i++)
    {
        idx[i] = i;
    }

    //descending sort
    sort(boxes.begin(), boxes.end(), std::greater<Anchor>());

    while(idx.size() > 0)
    {
        int good_idx = idx[0];
        filterOutBoxes.push_back(boxes[good_idx]);

        std::vector<size_t> tmp = idx;
        idx.clear();
        for(unsigned i = 1; i < tmp.size(); i++)
        {
            int tmp_i = tmp[i];
            float inter_x1 = std::max( boxes[good_idx][0], boxes[tmp_i][0] );
            float inter_y1 = std::max( boxes[good_idx][1], boxes[tmp_i][1] );
            float inter_x2 = std::min( boxes[good_idx][2], boxes[tmp_i][2] );
            float inter_y2 = std::min( boxes[good_idx][3], boxes[tmp_i][3] );

            float w = std::max((inter_x2 - inter_x1 + 1), 0.0F);
            float h = std::max((inter_y2 - inter_y1 + 1), 0.0F);

            float inter_area = w * h;
            float area_1 = (boxes[good_idx][2] - boxes[good_idx][0] + 1) * (boxes[good_idx][3] - boxes[good_idx][1] + 1);
            float area_2 = (boxes[tmp_i][2] - boxes[tmp_i][0] + 1) * (boxes[tmp_i][3] - boxes[tmp_i][1] + 1);
            float o = inter_area / (area_1 + area_2 - inter_area);           
            if( o <= threshold )
                idx.push_back(tmp_i);
        }
    }
}

AnchorGenerator::AnchorGenerator() {
}

AnchorGenerator::~AnchorGenerator() {
}

int AnchorGenerator::Init(int stride, const AnchorCfg& cfg, bool dense_anchor) {
	CRect2f base_anchor(0, 0, cfg.BASE_SIZE-1, cfg.BASE_SIZE-1);
	std::vector<CRect2f> ratio_anchors;
	// get ratio anchors
	_ratio_enum(base_anchor, cfg.RATIOS, ratio_anchors);
	_scale_enum(ratio_anchors, cfg.SCALES, preset_anchors);

	// save as x1,y1,x2,y2
	if (dense_anchor) {
		CHECK_EQ(stride % 2, 0);
		int num = preset_anchors.size();
		for (int i = 0; i < num; ++i) {
			CRect2f anchor = preset_anchors[i];
			preset_anchors.push_back(CRect2f(anchor[0]+int(stride/2),
									anchor[1]+int(stride/2),
									anchor[2]+int(stride/2),
									anchor[3]+int(stride/2)));
		}
	}

    anchor_stride = stride;

	anchor_num = preset_anchors.size();
    // for (int i = 0; i < anchor_num; ++i) {
        // preset_anchors[i].print();
    // }
	return anchor_num;
}

int AnchorGenerator::FilterAnchor(const float* cls_data,  const float* reg_data, const float* pts_data, const int h, const int w, std::vector<Anchor>& result) {

    int dim = h * w;
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            int id = i * w + j;
            for (int a = 0; a < anchor_num; ++a)
            {
                // foreground anchor?
                // LOG(INFO) << "cls_data: " << cls_data[(anchor_num + a) * dim + id];
                CHECK(cls_data[(anchor_num + a) * dim + id] > 0) << cls_data[(anchor_num + a) * dim + id];
                if (cls_data[(anchor_num + a) * dim + id] >= cls_threshold) {
                    // printf("cls %f\n", cls->channel(anchor_num + a)[id]);
                    CRect2f box(j * anchor_stride + preset_anchors[a][0],
                            i * anchor_stride + preset_anchors[a][1],
                            j * anchor_stride + preset_anchors[a][2],
                            i * anchor_stride + preset_anchors[a][3]);
                    // printf("%f %f %f %f\n", box[0], box[1], box[2], box[3]);
                    CRect2f delta(
                            reg_data[dim * a * 4 + id],
                            reg_data[(a * 4 + 1) * dim + id],
                            reg_data[(a * 4 + 2) * dim + id],
                            reg_data[(a * 4 + 3) * dim + id]);

                    Anchor res;
                    res.anchor[0] = box[0];
                    res.anchor[1] = box[1];
                    res.anchor[2] = box[2];
                    res.anchor[3] = box[3];
                    bbox_pred(box, delta, res.finalbox);
                    // printf("bbox pred\n");
                    res.score = cls_data[(anchor_num + a) * dim + id];


                    // if (1) {
                    std::vector<CPointf> pts_delta(5);
                    for (int p = 0; p < 5; ++p) {
                        pts_delta[p].x = pts_data[(a * 5 * 2 + p * 2) * dim + id];
                        pts_delta[p].y = pts_data[(a * 5 * 2 + p * 2 + 1) * dim + id];
                    }
                        // printf("ready landmark_pred\n");
                    landmark_pred(box, pts_delta, res.pts);
                        // printf("landmark_pred\n");
                    // }
                    result.push_back(res);
                }
            }
        }
    }

    
    return 0;
}

void AnchorGenerator::_ratio_enum(const CRect2f& anchor, const std::vector<float>& ratios, std::vector<CRect2f>& ratio_anchors) {
	float w = anchor[2] - anchor[0] + 1;	
	float h = anchor[3] - anchor[1] + 1;
	float x_ctr = anchor[0] + 0.5 * (w - 1);
	float y_ctr = anchor[1] + 0.5 * (h - 1);

	ratio_anchors.clear();
	float sz = w * h;
	for (int s = 0; s < ratios.size(); ++s) {
		float r = ratios[s];
		float size_ratios = sz / r;
		float ws = std::sqrt(size_ratios);
		float hs = ws * r;
		ratio_anchors.push_back(CRect2f(x_ctr - 0.5 * (ws - 1),
								y_ctr - 0.5 * (hs - 1),
								x_ctr + 0.5 * (ws - 1),
								y_ctr + 0.5 * (hs - 1)));
	}
}

void AnchorGenerator::_scale_enum(const std::vector<CRect2f>& ratio_anchor, const std::vector<float>& scales, std::vector<CRect2f>& scale_anchors) {
	scale_anchors.clear();
	for (int a = 0; a < ratio_anchor.size(); ++a) {
		CRect2f anchor = ratio_anchor[a];
		float w = anchor[2] - anchor[0] + 1;	
		float h = anchor[3] - anchor[1] + 1;
		float x_ctr = anchor[0] + 0.5 * (w - 1);
		float y_ctr = anchor[1] + 0.5 * (h - 1);

		for (int s = 0; s < scales.size(); ++s) {
			float ws = w * scales[s];
			float hs = h * scales[s];
			scale_anchors.push_back(CRect2f(x_ctr - 0.5 * (ws - 1),
								y_ctr - 0.5 * (hs - 1),
								x_ctr + 0.5 * (ws - 1),
								y_ctr + 0.5 * (hs - 1)));
		}
	}

}

void AnchorGenerator::bbox_pred(const CRect2f& anchor, const CRect2f& delta, float* box) {
	float w = anchor[2] - anchor[0] + 1;	
	float h = anchor[3] - anchor[1] + 1;
	float x_ctr = anchor[0] + 0.5 * (w - 1);
	float y_ctr = anchor[1] + 0.5 * (h - 1);

    float dx = delta[0];
    float dy = delta[1];
    float dw = delta[2];
    float dh = delta[3];

    float pred_ctr_x = dx * w + x_ctr; 
    float pred_ctr_y = dy * h + y_ctr;
    float pred_w = std::exp(dw) * w; 
    float pred_h = std::exp(dh) * h;

    box[0] = pred_ctr_x - 0.5 * (pred_w - 1.0);
    box[1] = pred_ctr_y - 0.5 * (pred_h - 1.0);
    box[2] = pred_ctr_x + 0.5 * (pred_w - 1.0);
    box[3] = pred_ctr_y + 0.5 * (pred_h - 1.0);
}

void AnchorGenerator::landmark_pred(const CRect2f anchor, const std::vector<CPointf>& delta, std::vector<CPointf>& pts) {
	float w = anchor[2] - anchor[0] + 1;	
	float h = anchor[3] - anchor[1] + 1;
	float x_ctr = anchor[0] + 0.5 * (w - 1);
	float y_ctr = anchor[1] + 0.5 * (h - 1);

    pts.resize(delta.size());
    for (int i = 0; i < delta.size(); ++i) {
        pts[i].x = delta[i].x*w + x_ctr;
        pts[i].y = delta[i].y*h + y_ctr;
    }
}

retinaface::retinaface() : tensorNet()
{
    ac_.resize(_feat_stride_fpn.size());
    for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
        int stride = _feat_stride_fpn[i];
        anchor_num_ = ac_[i].Init(stride, anchor_cfg[stride], false);
    }

    for (int i = 0; i < _feat_stride_fpn.size(); ++i) { 
        char clsname[100]; sprintf(clsname, "face_rpn_cls_prob_reshape_stride%d", _feat_stride_fpn[i]);
        char regname[100]; sprintf(regname, "face_rpn_bbox_pred_stride%d", _feat_stride_fpn[i]);
        char ptsname[100]; sprintf(ptsname, "face_rpn_landmark_pred_stride%d", _feat_stride_fpn[i]);
        std::string t1(clsname);
        std::string t2(regname);
        std::string t3(ptsname);
        m_output_names.push_back(t1);
        m_output_names.push_back(t2);
        m_output_names.push_back(t3);
    }
}

retinaface::~retinaface()
{
}

std::shared_ptr<retinaface> retinaface::Create( const char* prototxt_path, const char* model_path, 
                 uint32_t maxBatchSize,
                 precisionType precision,
                 deviceType device, bool allowGPUFallback, nvinfer1::IInt8Calibrator* calibrator) {
    std::shared_ptr<retinaface> net = std::make_shared<retinaface>();
    // net->init(prototxt_path, model_path)
    net->LoadNetwork(prototxt_path ,model_path, NULL, net->m_input_name, net->m_output_names, maxBatchSize, precision, device, allowGPUFallback, calibrator);
    return net;
}

int retinaface::Detect( float* input, uint32_t width, uint32_t height, std::vector<Anchor>& detections) {
    LOG(INFO) << "detect " << mOutputs.size();
    CHECK(m_output_names.size() == mOutputs.size());
    CHECK(m_output_names.size() == 9);
    //FIXME: do we really need this in unified memory?
    if (CUDA_FAILED(cudaPreImageNetRGB( (float4*) input, width, height, mInputCUDA, width, height, GetStream()))) {
        LOG(ERROR) << "cudaPreImageNetRGB";
        return -1;
    }
    inferenceBuffers[0] = mInputCUDA;
    for (int i = 0; i < mOutputs.size() ; i++) {
		const int outputIndex = mEngine->getBindingIndex(mOutputs[i].name.c_str());
        inferenceBuffers[outputIndex] = mOutputs[i].CUDA;
    }
  PROFILER_BEGIN(PROFILER_NETWORK);
	if( !mContext->execute(1, inferenceBuffers) )
	{
		printf(LOG_TRT "detectNet::Detect() -- failed to execute TensorRT context\n");
		return -1;
	}
  PROFILER_END(PROFILER_NETWORK);
	// process with TensorRT

    std::vector<Anchor> proposals;
    // int anchor_num = ac_.getAnchorNum();

    LOG(INFO) << "anchor_num:" << anchor_num_;
    int j = 0;
    for (int i = 0; i < mOutputs.size(); i = i + 3) { 

        LOG(INFO) << mOutputs[i].name << ":" << mOutputs[i].dims.d[0] << "," << mOutputs[i].dims.d[1] << "," << mOutputs[i].dims.d[2];
        LOG(INFO) << mOutputs[i + 1].name << ":" << mOutputs[i + 1].dims.d[0] << "," << mOutputs[i + 1].dims.d[1] << "," << mOutputs[i + 1].dims.d[2];
        LOG(INFO) << mOutputs[i + 2].name << ":" << mOutputs[i + 2].dims.d[0] << "," << mOutputs[i + 2].dims.d[1] << "," << mOutputs[i + 2].dims.d[2];

        CHECK_EQ(mOutputs[i].dims.d[0], anchor_num_ * 2);
        CHECK_EQ(mOutputs[i + 1].dims.d[0], anchor_num_ * 4);
        CHECK_EQ(mOutputs[i + 2].dims.d[0] % anchor_num_,  0);

        ac_[j++].FilterAnchor(mOutputs[i].CPU, mOutputs[i + 1].CPU, mOutputs[i + 2].CPU, mOutputs[i].dims.d[1], mOutputs[i].dims.d[2], proposals);

    }
    LOG(INFO) << "proposals: " << proposals.size();

    nms_cpu(proposals, nms_threshold, detections);

    return 0;
}
