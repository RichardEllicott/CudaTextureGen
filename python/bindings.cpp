#include <nanobind/nanobind.h>
#include "core_api.h"
#include <nanobind/ndarray.h>
#include "erosion.h"  // or better: declare in erosion.h

namespace nb = nanobind;

NB_MODULE(cuda_hello, m) {
    m.def("hello", []() { return "Hello from Python!"; });
    m.def("cuda_hello", []() { cuda_hello(); });

    m.def("erosion",
      [](nb::ndarray<float, nb::ndim<2>, nb::c_contig> arr, int steps) {
          int h = arr.shape(0);
          int w = arr.shape(1);
          run_erosion(arr.data(), w, h, steps);
      },
      nb::arg("arr"), nb::arg("steps"));

    
}
