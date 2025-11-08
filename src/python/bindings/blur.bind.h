#pragma once

#include "blur.cuh"
#include "python_helper.h"

namespace blur {

namespace nb = nanobind;

inline void bind(nb::module_ &m) {

    m.def("blur", [](nb::ndarray<float, nb::ndim<2>, nb::c_contig> array, float amount, bool wrap) {
        int height = array.shape(0);
        int width = array.shape(1);
        blur::blur(array.data(), width, height, amount, wrap); }, nb::arg("array"), nb::arg("amount"), nb::arg("wrap") = true);
}

} // namespace blur