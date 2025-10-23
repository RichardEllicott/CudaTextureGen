/*
 */
#pragma once

#include "noise_generator.cuh"
#include "python_helper.h"

namespace nb = nanobind;

inline void bind_noise_generator(nb::module_ &m) {

    auto ngd = nb::class_<noise_generator::NoiseGenerator>(m, "NoiseGenerator")
                   .def(nb::init<>())

    // bind get/sets
#define X(TYPE, NAME, DEFAULT_VAL) \
    .def_prop_rw(#NAME, &noise_generator::NoiseGenerator::get_##NAME, &noise_generator::NoiseGenerator::set_##NAME)
                       NOISE_GENERATOR_PARAMETERS
#undef X

                   .def("fill", [](noise_generator::NoiseGenerator &self, nb::ndarray<float> arr) { // fill an existing array with noise
                       if (arr.ndim() != 2)
                           throw std::runtime_error("Expected a 2D float32 array");

                       int h = arr.shape(0);
                       int w = arr.shape(1);
                       float *data = arr.data();
                       self.fill(data, w, h); // fill the array data

                   })

                   .def("generate", [](noise_generator::NoiseGenerator &self, int width, int height) { // return a new array with noise
                       auto arr = python_helper::get_numpy_float_array(height, width);
                       float *data = arr.data();
                       self.fill(data, width, height);
                       return arr;
                   })

        ;

    // Type enumerators
    nb::enum_<noise_generator::NoiseGenerator::Type>(ngd, "Type")

#define X(NAME) \
    .value(#NAME, noise_generator::NoiseGenerator::Type::NAME)
        NOISE_GENERATOR_TYPES
#undef X
            .export_values();
}
