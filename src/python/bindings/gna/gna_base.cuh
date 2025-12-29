/*


*/
#pragma once
// #include "template_d_base.cuh"
// #include "core.h"
// #include "template_e_base.cuh"
// #include <stdexcept>
// #include <unordered_set>
#include "template_macro_undef.h"
#include <nanobind/nanobind.h>
#include <unordered_map>

#include <memory> // shared_ptr
#include <string>
#include <unordered_map>

#include "cuda_types.cuh" // DeviceArray

// ================================================================ //
#define TEMPLATE_CLASS_NAME GNA_Base
#define TEMPLATE_NAMESPACE gna_base

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS \
    X(bool, _debug, false, "")    \
    X(int, test_int, false, "")   \
    X(float, test_float, false, "")

// (TYPE, DIMENSIONS, DIM3, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_DEVICE_ARRAYS \
    X(float, 2, 1, input, "")        \
    X(float, 2, 1, output, "")

// ================================================================ //

namespace TEMPLATE_NAMESPACE {

namespace nb = nanobind;

class TEMPLATE_CLASS_NAME {

  public:
    std::unordered_map<std::string, nb::object> vars;

    nb::object getattr(const std::string &key) const {
        auto it = vars.find(key);
        if (it == vars.end()) {
            // throw nb::attribute_error("Attribute not found: " + key);
            throw nb::attribute_error((std::string("Attribute not found: ") + key).c_str());
        }
        return it->second;
    }

    void setattr(const std::string &key, nb::object value) {
        vars[key] = value;
    }

#ifdef TEMPLATE_CLASS_DEVICE_ARRAYS
#define X(TYPE, DIMENSIONS, DIM3, NAME, DESCRIPTION) \
    std::shared_ptr<core::cuda::DeviceArray<TYPE, DIMENSIONS>> NAME;
    TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X
#endif

    void ensure_arrays() {
#ifdef TEMPLATE_CLASS_DEVICE_ARRAYS
#define X(TYPE, DIMENSIONS, DIM3, NAME, DESCRIPTION)                          \
    if (!NAME) {                                                              \
        NAME = std::make_shared<core::cuda::DeviceArray<TYPE, DIMENSIONS>>(); \
    }
        TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X
#endif
    }

    void process() {

        // if (!input) throw std::runtime_error("GNA_Base.input is not set");
        // auto &input_ref = *input;
        // if (input_ref.empty()) throw std::runtime_error("GNA_Base.input is empty");

        // if (!output) {
        //     output
        // }

        ensure_arrays();
        auto &input_ref = *input;
        auto &output_ref = *output;

        if (input_ref.empty()) throw std::runtime_error("GNA_Base.input is empty");

        output_ref.resize(input_ref.shape());



        // now use arr freely
        // arr.empty()
        // arr.data()
        // arr.size()
        // arr.upload(...)
        // arr.download(...)
    }
};

} // namespace TEMPLATE_NAMESPACE
