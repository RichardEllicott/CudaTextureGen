/*

nb_object_to_any and any_to_nb_object functions to convert from nanobind object to and from std::any class

this is used to ensure that std::any will not contain a nanobind object as we get down to the C++/CUDA only layer

it's sort of like a firewall really as it block dissalowed types, the main problem being python types being stored in std::any


⚠️ i might opt out of this pattern, which sucks as it was difficult to achive, however i could instead use constexpr properties() type pattern

*/
#pragma once

#include <any>
#include <nanobind/nanobind.h>

#include "core/cuda/device_array.cuh"
#include "core/ref.h"

namespace nanobind::helper {

namespace nb = nanobind;

// ================================================================================================================================
// [Non Template VEersion (kept for clarity)]
// --------------------------------------------------------------------------------------------------------------------------------

// // nb::object => std::any
// inline std::any nb_object_to_any(const nb::object &value) {
//     if (nb::isinstance<nb::int_>(value))
//         return std::any(nb::cast<int>(value));
//     if (nb::isinstance<nb::float_>(value))
//         return std::any(nb::cast<float>(value));
//     if (nb::isinstance<nb::bool_>(value))
//         return std::any(nb::cast<bool>(value));
//     if (nb::isinstance<nb::str>(value))
//         return std::any(nb::cast<std::string>(value));

//     throw nb::type_error("Unsupported Python type for property");
// }

// // std::any => nb::object
// inline nb::object any_to_nb_object(const std::any &v) {
//     if (v.type() == typeid(int))
//         return nb::cast(std::any_cast<int>(v));
//     if (v.type() == typeid(float))
//         return nb::cast(std::any_cast<float>(v));
//     if (v.type() == typeid(bool))
//         return nb::cast(std::any_cast<bool>(v));
//     if (v.type() == typeid(std::string))
//         return nb::cast(std::any_cast<std::string>(v));

//     throw nb::type_error("Unsupported C++ type in std::any");
// }

// ================================================================================================================================
// [Template Version]
// --------------------------------------------------------------------------------------------------------------------------------

// Type Traits + Policy Class Pattern

template <typename T>
struct nb_convert; // forward declaration

// [Optional Generic Trait]
template <typename T>
struct nb_convert {
    static bool match_py(const nb::object &o) { return nb::isinstance<T>(o); }
    static T from_py(const nb::object &o) { return nb::cast<T>(o); }
    static nb::object to_py(const T &v) { return nb::cast(v); }
};

// // [Specific Policy Classes Pattern]

// template <>
// struct nb_convert<int> {
//     static bool match_py(const nb::object &o) { return nb::isinstance<nb::int_>(o); }
//     static int from_py(const nb::object &o) {
//         // printf("TRIGGER nb_convert for int\n"); // proves this overrides the generic trait
//         return nb::cast<int>(o);
//     }
//     static nb::object to_py(const int &v) { return nb::cast(v); }
// };

// template <>
// struct nb_convert<float> {
//     static bool match_py(const nb::object &o) { return nb::isinstance<nb::float_>(o); }
//     static float from_py(const nb::object &o) { return nb::cast<float>(o); }
//     static nb::object to_py(const float &v) { return nb::cast(v); }
// };

// template <>
// struct nb_convert<bool> {
//     static bool match_py(const nb::object &o) { return nb::isinstance<nb::bool_>(o); }
//     static bool from_py(const nb::object &o) { return nb::cast<bool>(o); }
//     static nb::object to_py(const bool &v) { return nb::cast(v); }
// };

// template <>
// struct nb_convert<std::string> {
//     static bool match_py(const nb::object &o) { return nb::isinstance<nb::str>(o); }
//     static std::string from_py(const nb::object &o) { return nb::cast<std::string>(o); }
//     static nb::object to_py(const std::string &v) { return nb::cast(v); }
// };

// --------------------------------------------------------------------------------------------------------------------------------

template <typename... Ts>
std::any nb_object_to_any_typed(const nb::object &value) {
    std::any result;

    bool matched = ((nb_convert<Ts>::match_py(value)
                         ? (result = std::any(nb_convert<Ts>::from_py(value)), true)
                         : false) ||
                    ...);

    if (!matched)
        throw nb::type_error("Unsupported Python type for property");

    return result;
}

template <typename... Ts>
nb::object any_to_nb_object_typed(const std::any &v) {
    nb::object result;

    bool matched = ((v.type() == typeid(Ts)
                         ? (result = nb_convert<Ts>::to_py(std::any_cast<Ts>(v)), true)
                         : false) ||
                    ...);

    if (!matched)
        throw nb::type_error("Unsupported C++ type in std::any");

    return result;
}

// #define SUPPORTED_TYPES int, float, bool, std::string

using DeviceArrayFloat1D = core::Ref<core::cuda::DeviceArray<float, 1>>;
using DeviceArrayFloat2D = core::Ref<core::cuda::DeviceArray<float, 2>>;
using DeviceArrayFloat3D = core::Ref<core::cuda::DeviceArray<float, 3>>;

using DeviceArrayInt1D = core::Ref<core::cuda::DeviceArray<int, 1>>;
using DeviceArrayInt2D = core::Ref<core::cuda::DeviceArray<int, 2>>;
using DeviceArrayInt3D = core::Ref<core::cuda::DeviceArray<int, 3>>;

#define SUPPORTED_TYPES     \
    int,                    \
        float,              \
        bool,               \
        std::string,        \
        DeviceArrayFloat1D, \
        DeviceArrayFloat2D, \
        DeviceArrayFloat3D, \
        DeviceArrayInt1D,   \
        DeviceArrayInt2D,   \
        DeviceArrayInt3D

inline std::any nb_object_to_any(const nb::object &value) {
    return nb_object_to_any_typed<SUPPORTED_TYPES>(value);
}

inline nb::object any_to_nb_object(const std::any &v) {
    return any_to_nb_object_typed<SUPPORTED_TYPES>(v);
}

#undef SUPPORTED_TYPES

} // namespace nanobind::helper
