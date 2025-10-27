#pragma once

#include "python_helper.h"
#include "template_class2.cuh"


#define STRINGIFY(x) #x
#define EXPAND_AND_STRINGIFY(x) STRINGIFY(x)

namespace TEMPLATE_CLASS_NAMESPACE {

namespace nb = nanobind;

inline void bind(nb::module_ &m) {

    auto ngd = nb::class_<TEMPLATE_CLASS_NAME>(m, EXPAND_AND_STRINGIFY(TEMPLATE_CLASS_NAME)).def(nb::init<>());

// bind erosion parameters
#define X(TYPE, NAME, DEFAULT_VAL) \
    ngd.def_prop_rw(EXPAND_AND_STRINGIFY(NAME), &TEMPLATE_CLASS_NAME::get_##NAME, &TEMPLATE_CLASS_NAME::set_##NAME);
    TEMPLATE_CLASS_PARAMETERS
#undef X

    // bind maps so we can easily read/write to the maps
#define X(TYPE, NAME)                                                                                                                                  \
    auto get_##NAME = [](TEMPLATE_CLASS_NAME &self) { return python_helper::array2d_to_numpy_array(self.NAME); };                                      \
    auto set_##NAME = [](TEMPLATE_CLASS_NAME &self, nb::ndarray<float, nb::c_contig> arr) { self.NAME = python_helper::numpy_array_to_array2d(arr); }; \
    ngd.def_prop_rw(EXPAND_AND_STRINGIFY(NAME), get_##NAME, set_##NAME);
    TEMPLATE_CLASS_MAPS
#undef X

    // // run main erosion function
    // ngd.def("run_erosion", [](TEMPLATE_CLASS_NAME &self, nb::ndarray<float> arr) {
    //     if (arr.ndim() != 2)
    //         throw std::runtime_error("Input must be a 2D float32 array");

    //     // self.set_height(arr.shape(0));
    //     // self.set_width(arr.shape(1));
    //     // self.run_erosion(arr.data());
    // });

    // test function with c_contig
    ngd.def("run_erosion", [](TEMPLATE_CLASS_NAME &self, nb::ndarray<float, nb::c_contig> arr) {
        if (arr.ndim() != 2)
            throw std::runtime_error("Input must be a 2D float32 array");

        // self.set_height(arr.shape(0));
        // self.set_width(arr.shape(1));
        // self.run_erosion(arr.data());
    });


    ngd.def("process", [](TEMPLATE_CLASS_NAME &self) {
        self.process();
    });


}

} // namespace TEMPLATE_CLASS_NAMESPACE

// #undef TEMPLATE_CLASS_NAME
// #undef TEMPLATE_CLASS_NAMESPACE
// #undef TEMPLATE_CLASS_PARAMETERS
// #undef TEMPLATE_CLASS_MAPS
