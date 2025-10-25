/*
 */
#pragma once

#include "erosion2.cuh"
#include "python_helper.h"

// 📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜

#define TEMPLATE_CLASS_NAME Erosion2
#define TEMPLATE_CLASS_NAMESPACE erosion2

#define TEMPLATE_CLASS_PARAMETERS      \
    X(size_t, width, 256)              \
    X(size_t, height, 256)             \
    X(float, rain_rate, 0.01f)         \
    X(float, evaporation_rate, 0.005f) \
    X(float, erosion_rate, 0.01f)      \
    X(float, deposition_rate, 0.25f)   \
    X(float, slope_threshold, 0.1f)    \
    X(int, steps, 128)

#define TEMPLATE_CLASS_MAPS \
    X(float, height_map)    \
    X(float, water_map)     \
    X(float, sediment_map)

// 📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜

#define STRINGIFY(x) #x
#define EXPAND_AND_STRINGIFY(x) STRINGIFY(x)

namespace TEMPLATE_CLASS_NAMESPACE {

namespace nb = nanobind;

inline void bind(nb::module_ &m) {

    auto ngd = nb::class_<TEMPLATE_CLASS_NAME>(m, EXPAND_AND_STRINGIFY(TEMPLATE_CLASS_NAME)).def(nb::init<>());

    // bind par get/sets
#define X(TYPE, NAME, DEFAULT_VAL) \
    ngd.def_prop_rw(#NAME, &TEMPLATE_CLASS_NAME::get_##NAME, &TEMPLATE_CLASS_NAME::set_##NAME);
    TEMPLATE_CLASS_PARAMETERS
#undef X
    // bind map get/sets
#define X(TYPE, NAME) \
    ngd.def_prop_rw(#NAME, &TEMPLATE_CLASS_NAME::get_##NAME, &TEMPLATE_CLASS_NAME::set_##NAME);
    TEMPLATE_CLASS_MAPS
#undef X

    // 🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅
    ngd.def("run_erosion", [](TEMPLATE_CLASS_NAME &self, nb::ndarray<float> arr) {
        if (arr.ndim() != 2)
            throw std::runtime_error("Input must be a 2D float32 array");

        self.set_height(arr.shape(0));
        self.set_width(arr.shape(1));
        self.set_height_map(arr.data());
        self.process();
    });
    // 🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅🍅
}

} // namespace TEMPLATE_CLASS_NAMESPACE

// 📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜

#undef TEMPLATE_CLASS_NAME
#undef TEMPLATE_CLASS_NAMESPACE
#undef TEMPLATE_CLASS_PARAMETERS
#undef TEMPLATE_CLASS_MAPS
