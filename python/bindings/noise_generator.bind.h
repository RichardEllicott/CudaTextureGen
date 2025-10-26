#pragma once

#include "noise_generator.cuh"
#include "python_helper.h"

namespace noise_generator {

namespace nb = nanobind;

inline void bind(nb::module_ &m) {

    auto ngd = nb::class_<NoiseGenerator>(m, "NoiseGenerator").def(nb::init<>());

    // bind get/sets
#define X(TYPE, NAME, DEFAULT_VAL) \
    ngd.def_prop_rw(#NAME, &NoiseGenerator::get_##NAME, &NoiseGenerator::set_##NAME);
    NOISE_GENERATOR_PARAMETERS
#undef X

    ngd.def("fill", [](NoiseGenerator &self, nb::ndarray<float> arr) { // fill an existing array with noise
        if (arr.ndim() != 2)
            throw std::runtime_error("Expected a 2D float32 array");

        int h = arr.shape(0);
        int w = arr.shape(1);
        float *data = arr.data();
        self.fill(data, w, h); // fill the array data

    });

    ngd.def("generate", [](NoiseGenerator &self, int width, int height) { // return a new array with noise
        auto arr = python_helper::get_numpy_float_array(height, width);
        float *data = arr.data();
        self.fill(data, width, height);
        return arr;
    });

      // Type enumerators
    nb::enum_<NoiseGenerator::Type>(ngd, "Type")

#define X(NAME) \
    .value(#NAME, NoiseGenerator::Type::NAME)
        NOISE_GENERATOR_TYPES
#undef X
            .export_values();
}

} // namespace noise_generator