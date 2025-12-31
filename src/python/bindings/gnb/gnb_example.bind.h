#pragma once

#include "gnb/gnb_example.cuh"
#include "nanobind_helper.h"

#include <any>

namespace TEMPLATE_NAMESPACE {

namespace nb = nanobind;

//
//
//

inline std::any nb_object_to_any(const nb::object &value) {
    if (nb::isinstance<nb::int_>(value)) {
        return std::any(nb::cast<int>(value));
    }
    if (nb::isinstance<nb::float_>(value)) {
        return std::any(nb::cast<float>(value));
    }
    if (nb::isinstance<nb::bool_>(value)) {
        return std::any(nb::cast<bool>(value));
    }
    if (nb::isinstance<nb::str>(value)) {
        return std::any(nb::cast<std::string>(value));
    }

    throw nb::type_error("Unsupported Python type for property");
}

inline nb::object any_to_nb_object(const std::any &v) {
    if (v.type() == typeid(int))
        return nb::cast(std::any_cast<int>(v));
    if (v.type() == typeid(float))
        return nb::cast(std::any_cast<float>(v));
    if (v.type() == typeid(bool))
        return nb::cast(std::any_cast<bool>(v));
    if (v.type() == typeid(std::string))
        return nb::cast(std::any_cast<std::string>(v));

    throw nb::type_error("Unsupported C++ type in std::any");
}

//
//
//

inline void bind(nb::module_ &m) {

    // init standard object
    auto ngd = nb::class_<TEMPLATE_CLASS_NAME>(m, EXPAND_AND_STRINGIFY(TEMPLATE_CLASS_NAME)).def(nb::init<>());

    // ================================================================================================================================

    // bind Ref<DeviceArray>'s
#ifdef TEMPLATE_CLASS_DEVICE_ARRAYS
#define X(TYPE, DIMENSIONS, DIM3, NAME, DESCRIPTION) \
    ngd.def_rw(EXPAND_AND_STRINGIFY(NAME), &TEMPLATE_CLASS_NAME::NAME);
    TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X
#endif

    // ================================================================================================================================

    // // dynamic properties fallback to the has/get/set storage
    // nb::helper::bind_dynamic_properties<TEMPLATE_CLASS_NAME>(
    //     ngd,
    //     &TEMPLATE_CLASS_NAME::has_property,
    //     &TEMPLATE_CLASS_NAME::get_property,
    //     &TEMPLATE_CLASS_NAME::set_property);

    // ================================================================================================================================

    ngd.def("process", [](TEMPLATE_CLASS_NAME &self) {
        self.process();
    });
}

} // namespace TEMPLATE_NAMESPACE