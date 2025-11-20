#pragma once

#include "erosion8.cuh"
#include "nanobind_helper.h"


#define STRINGIFY(x) #x
#define EXPAND_AND_STRINGIFY(x) STRINGIFY(x)

namespace TEMPLATE_NAMESPACE {

namespace nb = nanobind;

inline void bind(nb::module_ &m) {

    auto ngd = nb::class_<TEMPLATE_CLASS_NAME>(m, EXPAND_AND_STRINGIFY(TEMPLATE_CLASS_NAME)).def(nb::init<>());

// bind pars
#ifdef TEMPLATE_CLASS_PARAMETERS
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    ngd.def_prop_rw(EXPAND_AND_STRINGIFY(NAME), &TEMPLATE_CLASS_NAME::get_##NAME, &TEMPLATE_CLASS_NAME::set_##NAME, DESCRIPTION);
    TEMPLATE_CLASS_PARAMETERS
#undef X
#endif

    // bind DeviceArray1D's
#ifdef TEMPLATE_CLASS_DEVICE_ARRAYS
#define X(TYPE, DIMENSION, NAME, DESCRIPTION)                                                                                                           \
    auto get_##NAME = [](TEMPLATE_CLASS_NAME &self) { return nb::helper::numpy::to_array(self.NAME); };                                        \
    auto set_##NAME = [](TEMPLATE_CLASS_NAME &self, nb::ndarray<TYPE, nb::c_contig> array) { nb::helper::numpy::to_device_array(array, self.NAME); }; \
    ngd.def_prop_rw(EXPAND_AND_STRINGIFY(NAME), get_##NAME, set_##NAME, DESCRIPTION);
    TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X
#endif

        // bind DeviceArray2D's
    #ifdef TEMPLATE_CLASS_DEVICE_ARRAY_2DS
    #define X(TYPE, NAME, DESCRIPTION)                                                                                                                      \
    auto get_##NAME = [](TEMPLATE_CLASS_NAME &self) { return nb::helper::numpy::to_array(self.NAME); };                                        \
    auto set_##NAME = [](TEMPLATE_CLASS_NAME &self, nb::ndarray<TYPE, nb::c_contig> array) { nb::helper::numpy::to_device_array(array, self.NAME); }; \
    ngd.def_prop_rw(EXPAND_AND_STRINGIFY(NAME), get_##NAME, set_##NAME, DESCRIPTION);
        TEMPLATE_CLASS_DEVICE_ARRAY_2DS
    #undef X
    #endif

//     // ⚠️ WORKING support for None... but not using due to introspection loss! (however good example)
// #ifdef TEMPLATE_CLASS_DEVICE_ARRAY_2DS
// #define X(TYPE, NAME, DESCRIPTION)                                                                                              \
//     auto get_##NAME = [](TEMPLATE_CLASS_NAME &self) { return nb::helper::device_array_to_python(self.NAME); };               \
//     auto set_##NAME = [](TEMPLATE_CLASS_NAME &self, nb::object obj) { nb::helper::python_to_device_array(obj, self.NAME); }; \
//     ngd.def_prop_rw(EXPAND_AND_STRINGIFY(NAME), get_##NAME, set_##NAME, DESCRIPTION, nb::arg("value").none(true));
//     TEMPLATE_CLASS_DEVICE_ARRAY_2DS
// #undef X
// #endif

    // bind DeviceArray3D's
#ifdef TEMPLATE_CLASS_DEVICE_ARRAY_3DS
#define X(TYPE, NAME, DESCRIPTION)                                                                                                                       \
    auto get_##NAME = [](TEMPLATE_CLASS_NAME &self) { return nb::helper::numpy::to_array(self.NAME); };                                         \
    auto set_##NAME = [](TEMPLATE_CLASS_NAME &self, nb::ndarray<float, nb::c_contig> array) { nb::helper::numpy::to_device_array(array, self.NAME); }; \
    ngd.def_prop_rw(EXPAND_AND_STRINGIFY(NAME), get_##NAME, set_##NAME, DESCRIPTION);
    TEMPLATE_CLASS_DEVICE_ARRAY_3DS
#undef X
#endif

// Method's  (⚠️ Experimental)
#ifdef TEMPLATE_CLASS_METHODS
#define X(TYPE, NAME)                                                   \
    ngd.def(EXPAND_AND_STRINGIFY(NAME), [](TEMPLATE_CLASS_NAME &self) { \
        self.NAME();                                                    \
    });
    TEMPLATE_CLASS_METHODS
#undef X
#endif

    ngd.def("deallocate_device", [](TEMPLATE_CLASS_NAME &self) {
        self.deallocate_device();
    });

    ngd.def("allocate_device", [](TEMPLATE_CLASS_NAME &self) {
        self.allocate_device();
    });

    ngd.def("process", [](TEMPLATE_CLASS_NAME &self) {
        self.process();
    });
}

} // namespace TEMPLATE_NAMESPACE

#include "template_macro_undef.h"
