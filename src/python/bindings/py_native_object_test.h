/*

testing a native numpy object

*/
#pragma once

// ════════════════════════════════════════════════ //
#define TEMPLATE_CLASS_NAME PyNativeObjectTest
#define TEMPLATE_NAMESPACE py_native_object_test

#define TEMPLATE_CLASS_PARAMETERS    \
    X(size_t, _block, 16)            \
    X(size_t, width, 256)            \
    X(size_t, height, 256)           \
    X(int, mode, 0)                  \
    X(int, steps, 1024)              \
    X(float, rain_rate, 0.01)        \
    X(bool, wrap, true)              \
    X(float, max_water_outflow, 1.0) \
    X(float, capacity, 0.1)          \
    X(float, erode, 0.1)             \
    X(float, deposit, 0.1)           \
    X(float, evaporation_rate, 0.1)  \
    X(bool, debug_hash_cell_order, false)
// ════════════════════════════════════════════════ //

#define STRINGIFY(x) #x
#define EXPAND_AND_STRINGIFY(x) STRINGIFY(x)

#include "nanobind_helper.h"

namespace TEMPLATE_NAMESPACE {

namespace nb = nanobind;

class TEMPLATE_CLASS_NAME {
  private:
    nb::ndarray<nb::numpy, float> _array; // or double/int, etc.

    float _test;

  public:
    TEMPLATE_CLASS_NAME(nb::ndarray<nb::numpy, float> arr) : _array(arr) {}

    void process() {
        // Example: access shape
        // auto shape = arr_.shape();
        // shape[0], shape[1], ...
    }

    float get_test() {
        return _test;
    }

    void set_test(float value) {
        _test = value;
    }

    static void bind(nb::module_ &m) {

        auto ngd = nb::class_<TEMPLATE_CLASS_NAME>(m, EXPAND_AND_STRINGIFY(TEMPLATE_CLASS_NAME));

        ngd.def(nb::init<nb::ndarray<nb::numpy, float>>());

        // nb::init<nb::ndarray<nb::numpy, float, nb::c_contig>>() // enforce c_contig

        ngd.def("process", &TEMPLATE_CLASS_NAME::process);

        // ngd.def_prop_rw("test", &TEMPLATE_CLASS_NAME::get_test, &TEMPLATE_CLASS_NAME::set_test);

        ngd.def_rw("test", &TEMPLATE_CLASS_NAME::_test);
        ngd.def_rw("array", &TEMPLATE_CLASS_NAME::_array);
    }
};

/*

import numpy as np
import py_native_object_test

arr = np.ones((4,4), dtype=np.float32)
obj = py_native_object_test.PyNativeObjectTest(arr)
obj.process()

*/

} // namespace TEMPLATE_NAMESPACE

#undef TEMPLATE_CLASS_NAME
#undef TEMPLATE_NAMESPACE
#undef TEMPLATE_CLASS_PARAMETERS
#undef TEMPLATE_CLASS_MAPS
#undef TEMPLATE_CLASS_MAPS2
#undef TEMPLATE_CLASS_TYPES
#undef STRINGIFY
#undef EXPAND_AND_STRINGIFY