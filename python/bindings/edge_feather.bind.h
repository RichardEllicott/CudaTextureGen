/*
 */
#pragma once

#include "edge_feather.cuh"
#include "python_helper.h"

namespace edge_feather {

namespace nb = nanobind;

inline void bind(nb::module_ &m) {

    // m.def("edge_feather", [](nb::ndarray<float, nb::ndim<2>, nb::c_contig> arr, float amount, bool wrap) {
    // int height = arr.shape(0);
    // int wwidth = arr.shape(1);
    // blur::blur(arr.data(), w, h, amount, wrap); }, nb::arg("arr"), nb::arg("amount"), nb::arg("wrap") = false);
    // });
}

} // namespace edge_feather