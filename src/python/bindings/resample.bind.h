/*
 */
#pragma once

#include "nanobind_helper.h"
#include "resample.cuh"

#define STRINGIFY(x) #x
#define EXPAND_AND_STRINGIFY(x) STRINGIFY(x)

namespace resample {

namespace nb = nanobind;

inline void bind(nb::module_ &m) {

    auto ngd = nb::class_<Resample>(m, "Resample").def(nb::init<>());

    // bind erosion parameters
#define X(TYPE, NAME, DEFAULT_VAL) \
    ngd.def_prop_rw(EXPAND_AND_STRINGIFY(NAME), &Resample::get_##NAME, &Resample::set_##NAME);
    RESAMPLE_PARAMETERS
#undef X

    // bind maps
#define X(TYPE, NAME)                                                                                                                                                \
    auto get_##NAME = [](Resample &self) { return nb::helper::numpy::to_array(self.NAME); };                                                               \
    auto set_##NAME = [](Resample &self, nb::ndarray<float, nb::c_contig> arr) { self.NAME = core::cuda::CudaArray2D<TYPE>(nb::helper::numpy::to_array_2d(arr)); }; \
    ngd.def_prop_rw(EXPAND_AND_STRINGIFY(NAME), get_##NAME, set_##NAME);
    RESAMPLE_MAPS
#undef X

    ngd.def("process", [](Resample &self) {
        self.process();
    });
}

} // namespace resample


// #undef TEMPLATE_CLASS_NAME
// #undef TEMPLATE_NAMESPACE
#undef RESAMPLE_PARAMETERS
#undef RESAMPLE_MAPS

#undef STRINGIFY
#undef EXPAND_AND_STRINGIFY
