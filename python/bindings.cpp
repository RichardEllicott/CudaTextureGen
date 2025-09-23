#include <nanobind/nanobind.h>
#include "core_api.h"

namespace nb = nanobind;

NB_MODULE(cuda_hello, m) {
    m.def("hello", []() { return "Hello from Python!"; });
    m.def("cuda_hello", []() { cuda_hello(); });
}
