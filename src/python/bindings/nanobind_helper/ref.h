/*
⚠️ NOT WORKING, NOT SURE WHERE TO PUT THIS YET... still have a copy in bindings.cpp

allows nanobind to understand the custom core::Ref which is a custom wrapper for a shared_ptr

this is not to allow Python to see the Ref exactly, but just handle it like a shared_ptr

*/
// #pragma once

// #include "core/ref.h"
// #include <nanobind/nanobind.h>
// #include <nanobind/stl/shared_ptr.h>

// namespace nb = nanobind;

// template <typename T>
// struct nb::detail::type_caster<core::Ref<T>> {
//     using Value = core::Ref<T>;
//     using Inner = std::shared_ptr<T>;

//     NB_TYPE_CASTER(Value, const_name("Ref[") + make_caster<T>::Name + const_name("]"));

//     // Python → C++
//     bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
//         make_caster<Inner> inner_caster;
//         if (!inner_caster.from_python(src, flags, cleanup))
//             return false;

//         value.shared_ptr = cast<Inner>(src);
//         return true;
//     }

//     // C++ → Python
//     static handle from_cpp(const Value &v, rv_policy policy, cleanup_list *cleanup) noexcept {
//         return make_caster<Inner>::from_cpp(v.shared_ptr, policy, cleanup);
//     }
// };