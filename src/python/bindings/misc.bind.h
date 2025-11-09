#pragma once

// #include "crater_imprint.cuh"
#include "python_helper.h"

namespace misc {

namespace nb = nanobind;

inline void bind(nb::module_ &m) {

    m.def("test123", [](nb::ndarray<float, nb::ndim<2>, nb::c_contig> array, float amount, bool wrap) {
        int h = array.shape(0);
        int w = array.shape(1);
        // blur::blur(arr.data(), w, h, amount, wrap);

        




    },
          nb::arg("array"), nb::arg("amount"), nb::arg("wrap") = true);
}

} // namespace misc