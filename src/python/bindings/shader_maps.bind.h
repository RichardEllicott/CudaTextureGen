#pragma once

#include "python_helper.h"
#include "shader_maps.cuh"

namespace shader_maps {

namespace nb = nanobind;

inline void bind(nb::module_ &m) {

    // standalone generate normal map
    m.def("generate_normal_map", [](nb::ndarray<float> array, float strength, bool wrap) {
        if (array.ndim() != 2)
            throw std::runtime_error("Expected a 2D float32 array");
        int height = array.shape(0);
        int width = array.shape(1);
        auto normal_array = python_helper::get_numpy_float_array(height, width, 3); // 3D numpy array (rgb)
        generate_normal_map(array.data(), normal_array.data(), width, height, strength, wrap);
        return normal_array; }, nb::arg("array"), nb::arg("strength") = 1.0, nb::arg("wrap") = true);

    // standalone generate ambient occlusion map
    m.def("generate_ao_map", [](nb::ndarray<float> array, float radius, bool wrap, int mode) {
        if (array.ndim() != 2)
            throw std::runtime_error("Expected a 2D float32 array");

        int height = array.shape(0);
        int width = array.shape(1);
        auto ao_array = python_helper::get_numpy_float_array(height, width); // 3D numpy array (rgb)
        generate_ao_map(array.data(), ao_array.data(), width, height, radius, wrap, mode);
        return ao_array; }, nb::arg("array"), nb::arg("radius") = 1.0f, nb::arg("wrap") = true, nb::arg("mode") = 0);
}

} // namespace shader_maps