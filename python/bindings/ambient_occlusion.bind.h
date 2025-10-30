#pragma once

#include "ambient_occlusion.cuh"
#include "python_helper.h"

namespace ambient_occlusion {

namespace nb = nanobind;

inline void bind(nb::module_ &m) {

//     m.def("blur", [](nb::ndarray<float, nb::ndim<2>, nb::c_contig> arr, float amount, bool wrap) {
//         int h = arr.shape(0);
//         int w = arr.shape(1);
//         blur::blur(arr.data(), w, h, amount, wrap); }, nb::arg("arr"), nb::arg("amount"), nb::arg("wrap") = true);
}

} // namespace blur