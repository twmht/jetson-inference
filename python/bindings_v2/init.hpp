#pragma once

#include <pybind11/pybind11.h>
#include "tensorNet.h"
#include "retinaface.h"
#include "EntropyCalibrator.h"
namespace py = pybind11;
std::shared_ptr<retinaface> createRetinaFace(const char* prototxt_path, const char* model_path, 
                 uint32_t maxBatchSize,
                 precisionType precision,
                 deviceType device, bool allowGPUFallback, const std::string cal_table);
void init_tensornet(py::module &);
