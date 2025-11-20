#pragma once

#include "blur.cuh"
#include "nanobind_helper.h"

namespace blur {

namespace nb = nanobind;

inline void bind(nb::module_ &m) {

    // ⚠️ using this pattern as the other pattern silently converts the array to contig
    // this one will throw an error with non c_contiguous
    m.def("blur", [](nb::ndarray<float> array, float amount, bool wrap) {
    if (!nb::helper::numpy::is_c_contiguous(array)) {
        throw std::runtime_error("blur: array must be C-contiguous");
    }
    
    if (array.ndim() != 2 && array.ndim() != 3) {
        throw std::runtime_error("blur: expected 2D or 3D array");
    }
    
    if (array.ndim() == 2) {
        int h = array.shape(0);
        int w = array.shape(1);
        blur::blur(array.data(), w, h, amount, wrap);
    } else {
        int h = array.shape(0);
        int w = array.shape(1);
        int c = array.shape(2);
        blur::blur(array.data(), w, h, c, amount, wrap);
    } }, nb::arg("array"), nb::arg("amount"), nb::arg("wrap") = true);
}
} // namespace blur