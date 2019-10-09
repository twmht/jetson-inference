#include "init.hpp"
#include <iostream>
#include <stdexcept>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

std::shared_ptr<retinaface> createRetinaFace(const char* prototxt_path, const char* model_path, 
                 uint32_t maxBatchSize,
                 precisionType precision,
                 deviceType device, bool allowGPUFallback, const std::string cal_table) {

  std::unique_ptr<nvinfer1::IInt8Calibrator> calibrator(nullptr);
  if (cal_table != "") {
    calibrator.reset(new Int8EntropyCalibrator2(cal_table));
  }
  std::shared_ptr<retinaface> net = retinaface::Create(prototxt_path, model_path, maxBatchSize, precision, device, allowGPUFallback, calibrator.get());
  return net;
}

PYBIND11_MODULE(pyinference, m) {
  init_tensornet(m);
  py::enum_<deviceType>(m, "deviceType")
      .value("DEVICE_GPU", DEVICE_GPU)
      .export_values();

  py::enum_<precisionType>(m, "precisionType")
      .value("TYPE_FASTEST", TYPE_FASTEST)
      .value("TYPE_FP32", TYPE_FP32)
      .value("TYPE_FP16", TYPE_FP16)
      .value("TYPE_INT8", TYPE_INT8)
      .export_values();
  m.def("create_retinaface", &createRetinaFace);
}

