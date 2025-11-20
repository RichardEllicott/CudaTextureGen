#pragma once

#include "nanobind_helper.h"
#include "template_darray_1.cuh"

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
    auto get_##NAME = [](TEMPLATE_CLASS_NAME &self) { return nb::helper::device_array_to_numpy(self.NAME); };                                        \
    auto set_##NAME = [](TEMPLATE_CLASS_NAME &self, nb::ndarray<TYPE, nb::c_contig> array) { nb::helper::numpy_to_device_array(array, self.NAME); }; \
    ngd.def_prop_rw(EXPAND_AND_STRINGIFY(NAME), get_##NAME, set_##NAME, DESCRIPTION);
    TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X
#endif

    // bind DeviceArray2D's
#ifdef TEMPLATE_CLASS_DEVICE_ARRAY_2DS
#define X(TYPE, NAME, DESCRIPTION)                                                                                                                      \
    auto get_##NAME = [](TEMPLATE_CLASS_NAME &self) { return nb::helper::device_array_to_numpy(self.NAME); };                                        \
    auto set_##NAME = [](TEMPLATE_CLASS_NAME &self, nb::ndarray<TYPE, nb::c_contig> array) { nb::helper::numpy_to_device_array(array, self.NAME); }; \
    ngd.def_prop_rw(EXPAND_AND_STRINGIFY(NAME), get_##NAME, set_##NAME, DESCRIPTION);
    TEMPLATE_CLASS_DEVICE_ARRAY_2DS
#undef X
#endif

    // bind DeviceArray3D's
#ifdef TEMPLATE_CLASS_DEVICE_ARRAY_3DS
#define X(TYPE, NAME, DESCRIPTION)                                                                                                                       \
    auto get_##NAME = [](TEMPLATE_CLASS_NAME &self) { return nb::helper::device_array_to_numpy(self.NAME); };                                         \
    auto set_##NAME = [](TEMPLATE_CLASS_NAME &self, nb::ndarray<float, nb::c_contig> array) { nb::helper::numpy_to_device_array(array, self.NAME); }; \
    ngd.def_prop_rw(EXPAND_AND_STRINGIFY(NAME), get_##NAME, set_##NAME, DESCRIPTION);
    TEMPLATE_CLASS_DEVICE_ARRAY_3DS
#undef X
#endif

// Method's  (⚠️ Experimental)
#ifdef TEMPLATE_CLASS_METHODS
#define X(TYPE, NAME) \
     ngd.def(EXPAND_AND_STRINGIFY(NAME), [](TEMPLATE_CLASS_NAME &self) {\
        self.NAME();\
     });
    TEMPLATE_CLASS_METHODS
#undef X
#endif



    ngd.def("process", [](TEMPLATE_CLASS_NAME &self) {
        self.process();
    });
}

} // namespace TEMPLATE_NAMESPACE

#include "template_macro_undef.h"
