#pragma once

#include "erosion3.cuh"
#include "python_helper.h"

#define STRINGIFY(x) #x
#define EXPAND_AND_STRINGIFY(x) STRINGIFY(x)

namespace TEMPLATE_NAMESPACE {

namespace nb = nanobind;

inline void bind(nb::module_ &m) {

    auto ngd = nb::class_<TEMPLATE_CLASS_NAME>(m, EXPAND_AND_STRINGIFY(TEMPLATE_CLASS_NAME)).def(nb::init<>());

    // bind erosion parameters
#define X(TYPE, NAME, DEFAULT_VAL) \
    ngd.def_prop_rw(EXPAND_AND_STRINGIFY(NAME), &TEMPLATE_CLASS_NAME::get_##NAME, &TEMPLATE_CLASS_NAME::set_##NAME);
    TEMPLATE_CLASS_PARAMETERS
#undef X

    // bind maps
#define X(TYPE, NAME)                                                                                                                                                           \
    auto get_##NAME = [](TEMPLATE_CLASS_NAME &self) { return python_helper::array2d_to_numpy_array(self.NAME); };                                                               \
    auto set_##NAME = [](TEMPLATE_CLASS_NAME &self, nb::ndarray<float, nb::c_contig> arr) { self.NAME = core::CudaArray2D<TYPE>(python_helper::numpy_array_to_array2d(arr)); }; \
    ngd.def_prop_rw(EXPAND_AND_STRINGIFY(NAME), get_##NAME, set_##NAME);
    TEMPLATE_CLASS_MAPS
#undef X

    //
    //

// 🚧 bind Type enumerators (new pattern?? we could check for a def)
#ifdef TEMPLATE_CLASS_TYPES
    nb::enum_<TEMPLATE_CLASS_NAME::Type>(ngd, "Type")
#define X(NAME) \
    .value(#NAME, TEMPLATE_CLASS_NAME::Type::NAME)
        TEMPLATE_CLASS_TYPES
#undef X
            .export_values();
#endif
    //
    //
    //

    ngd.def("process", [](TEMPLATE_CLASS_NAME &self) {
        self.process();
    });
}

} // namespace TEMPLATE_NAMESPACE

#undef TEMPLATE_CLASS_NAME
#undef TEMPLATE_NAMESPACE
#undef TEMPLATE_CLASS_PARAMETERS
#undef TEMPLATE_CLASS_MAPS
#undef STRINGIFY
#undef EXPAND_AND_STRINGIFY


