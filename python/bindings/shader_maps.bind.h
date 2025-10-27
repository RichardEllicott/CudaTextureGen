/*
 */
#pragma once

#include "python_helper.h"
#include "shader_maps.cuh"

namespace shader_maps {

namespace nb = nanobind;

inline void bind(nb::module_ &m) {

    auto ngd = nb::class_<shader_maps::ShaderMaps>(m, "ShaderMaps").def(nb::init<>()); // init

    // bind generate_normal_map
    ngd.def("generate_normal_map", [](shader_maps::ShaderMaps &self, nb::ndarray<float> arr, float amount, bool wrap) {
        if (arr.ndim() != 2)
            throw std::runtime_error("Expected a 2D float32 array");

        int height = arr.shape(0);
        int width = arr.shape(1);
        auto normal_arr = python_helper::get_numpy_float_array(height, width, 3); // 3D numpy array (rgb)
        self.generate_normal_map(arr.data(), normal_arr.data(), width, height, amount, wrap);
        return normal_arr; // ret
    },
            nb::arg("arr"), nb::arg("amount") = 1.0f, nb::arg("wrap") = true); // defaults

    // bind generate_ao_map
    ngd.def("generate_ao_map", [](shader_maps::ShaderMaps &self, nb::ndarray<float> arr, float radius = 1.0f, bool wrap = true) {
        if (arr.ndim() != 2)
            throw std::runtime_error("Expected a 2D float32 array");

        int height = arr.shape(0);
        int width = arr.shape(1);
        auto ao_arr = python_helper::get_numpy_float_array(height, width); // 3D numpy array (rgb)
        self.generate_ao_map(arr.data(), ao_arr.data(), width, height, 1.0f, true);
        return ao_arr; // ret
    },
            nb::arg("arr"), nb::arg("radius") = 1.0f, nb::arg("wrap") = true) // defaults
        ;

    //  nb::arg("self"), nb::arg("arr"), nb::arg("radius") = 1.0f, nb::arg("wrap") = true
}

} // namespace shader_maps