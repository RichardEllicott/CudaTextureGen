/*

type_caster

allows nanobind to understand the custom core::Ref which is a custom wrapper for a shared_ptr

this is not to allow Python to see the Ref exactly, but just handle it like a shared_ptr

*/
#pragma once

#include <cuda_runtime.h> // defines float2
#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>

namespace nb = nanobind;

template <>
struct nb::detail::type_caster<float2> {
    NB_TYPE_CASTER(float2, const_name("float2"));

    // Python → C++
    bool from_python(nb::handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
        if (!nb::isinstance<nb::sequence>(src))
            return false;

        nb::sequence seq = nb::borrow<nb::sequence>(src);
        if (nb::len(seq) != 2)
            return false;

        value.x = nb::cast<float>(seq[0]);
        value.y = nb::cast<float>(seq[1]);
        return true;
    }

    // C++ → Python
    static nb::handle from_cpp(const float2 &v,
                               nb::rv_policy policy,
                               nb::detail::cleanup_list *cleanup) noexcept {
        return nb::make_tuple(v.x, v.y).release();
    }
};