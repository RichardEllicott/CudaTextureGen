#pragma once

#include "fluid_simulation.cuh"
#include "python_helper.h"

#define STRINGIFY(x) #x
#define EXPAND_AND_STRINGIFY(x) STRINGIFY(x)

namespace python_helper {

} // namespace python_helper

namespace TEMPLATE_NAMESPACE {

namespace nb = nanobind;

inline void bind(nb::module_ &m) {

    auto ngd = nb::class_<TEMPLATE_CLASS_NAME>(m, EXPAND_AND_STRINGIFY(TEMPLATE_CLASS_NAME)).def(nb::init<>());

    // bind erosion parameters
#ifdef TEMPLATE_CLASS_PARAMETERS
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    ngd.def_prop_rw(EXPAND_AND_STRINGIFY(NAME), &TEMPLATE_CLASS_NAME::get_##NAME, &TEMPLATE_CLASS_NAME::set_##NAME);
    TEMPLATE_CLASS_PARAMETERS
#undef X
#endif

    // bind maps
#ifdef TEMPLATE_CLASS_MAPS
#define X(TYPE, NAME, DESCRIPTION)                                                                                                                                                    \
    auto get_##NAME = [](TEMPLATE_CLASS_NAME &self) { return python_helper::array2d_to_numpy_array(self.NAME); };                                                                     \
    auto set_##NAME = [](TEMPLATE_CLASS_NAME &self, nb::ndarray<float, nb::c_contig> arr) { self.NAME = core::cuda::CudaArray2D<TYPE>(python_helper::numpy_array_to_array2d(arr)); }; \
    ngd.def_prop_rw(EXPAND_AND_STRINGIFY(NAME), get_##NAME, set_##NAME);
    TEMPLATE_CLASS_MAPS
#undef X
#endif

#ifdef TEMPLATE_CLASS_TYPES
    nb::enum_<TEMPLATE_CLASS_NAME::Type>(ngd, "Type")

#define X(NAME, DESCRIPTION) \
    .value(#NAME, TEMPLATE_CLASS_NAME::Type::NAME)
        TEMPLATE_CLASS_TYPES
#undef X
            .export_values();
#endif

    // default process function
    ngd.def("process", [](TEMPLATE_CLASS_NAME &self) {
        self.process();

        // return python_helper::array2d_to_numpy_array(self.image); // optional return array
    });

    //
    //


    // NEW DeviceArray2D hooks
#ifdef TEMPLATE_CLASS_DEVICE_ARRAY_2DS
#define X(TYPE, NAME, DESCRIPTION)                                                                                                                          \
    auto get_##NAME = [](TEMPLATE_CLASS_NAME &self) { return python_helper::device_array_2d_to_numpy(self.NAME); };                                         \
    auto set_##NAME = [](TEMPLATE_CLASS_NAME &self, nb::ndarray<float, nb::c_contig> array) { python_helper::numpy_to_device_array_2d(array, self.NAME); }; \
    ngd.def_prop_rw(EXPAND_AND_STRINGIFY(NAME), get_##NAME, set_##NAME);
    TEMPLATE_CLASS_DEVICE_ARRAY_2DS
#undef X
#endif

    //
    //
    //
    //
}

} // namespace TEMPLATE_NAMESPACE

#include "template_macro_undef.h"