#include <iostream>
#include "init.hpp"
#include "tensorNet.h"


void init_tensornet(py::module & m) {
  py::class_<tensorNet>(m, "tensorNet");
}

