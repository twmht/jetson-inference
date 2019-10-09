#include <iostream>
#include "init.hpp"
#include "retinaface.h"

// static std::shared_ptr<retinaface> Create( const char* prototxt_path, const char* model_path, 
         // uint32_t maxBatchSize=1, 
         // precisionType precision=TYPE_FASTEST,
            // deviceType device=DEVICE_GPU, bool allowGPUFallback=true, nvinfer1::IInt8Calibrator* calibrator=NULL);
void init_retinaface(py::module & m) {
  py::class_<retinaface>(m, "retinaface");
    // .def_static("create", static_cast<void >);
}
