#pragma once

#include "python_helper.h"
#include "template_class.cuh"

// 📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜

#define TEMPLATE_CLASS_NAME TemplateClass
#define TEMPLATE_CLASS_NAMESPACE template_class

#define TEMPLATE_CLASS_PARAMETERS \
    X(size_t, width, 256)         \
    X(size_t, height, 256)        \
    X(float, test_par1, 0.0)      \
    X(float, test_par2, 1.0)      \
    X(float, test_par3, 1.0)

#define TEMPLATE_CLASS_MAPS \
    X(float, height_map)    \
    X(float, blend_mask)    \
    X(float, gradient_map)

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
    ngd.def("test", [](TEMPLATE_CLASS_NAME &self, nb::ndarray<float> arr) {
        if (arr.ndim() != 2)
            throw std::runtime_error("Input must be a 2D float32 array");

        int height = arr.shape(0);
        int width = arr.shape(1);
        float *data = arr.data();

        self.set_height(height);
        self.set_width(width);
        self.set_height_map(data);
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
