/*

type_caster

allows nanobind to understand the custom core::Ref which is a custom wrapper for a shared_ptr

this is not to allow Python to see the Ref exactly, but just handle it like a shared_ptr

*/
#pragma once

#define CODE_ROUTE 1 // macro version

#if CODE_ROUTE == 0

// ================================================================================================================================
// WORKING PLAIN float2 version
// --------------------------------------------------------------------------------------------------------------------------------

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

#endif

#if CODE_ROUTE == 1

// ================================================================================================================================
// Claude MACRO version
// --------------------------------------------------------------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>

namespace nb = nanobind;

// Macro for 2-component vector types
#define NB_TYPE_CASTER_VEC2(VecType, ElemType, type_name)                                 \
    template <>                                                                           \
    struct nb::detail::type_caster<VecType> {                                             \
        NB_TYPE_CASTER(VecType, const_name(type_name));                                   \
                                                                                          \
        bool from_python(nb::handle src, uint8_t flags, cleanup_list *cleanup) noexcept { \
            if (!nb::isinstance<nb::sequence>(src))                                       \
                return false;                                                             \
            nb::sequence seq = nb::borrow<nb::sequence>(src);                             \
            if (nb::len(seq) != 2)                                                        \
                return false;                                                             \
            value.x = nb::cast<ElemType>(seq[0]);                                         \
            value.y = nb::cast<ElemType>(seq[1]);                                         \
            return true;                                                                  \
        }                                                                                 \
                                                                                          \
        static nb::handle from_cpp(const VecType &v, nb::rv_policy policy,                \
                                   nb::detail::cleanup_list *cleanup) noexcept {          \
            return nb::make_tuple(v.x, v.y).release();                                    \
        }                                                                                 \
    };

// Macro for 3-component vector types
#define NB_TYPE_CASTER_VEC3(VecType, ElemType, type_name)                                 \
    template <>                                                                           \
    struct nb::detail::type_caster<VecType> {                                             \
        NB_TYPE_CASTER(VecType, const_name(type_name));                                   \
                                                                                          \
        bool from_python(nb::handle src, uint8_t flags, cleanup_list *cleanup) noexcept { \
            if (!nb::isinstance<nb::sequence>(src))                                       \
                return false;                                                             \
            nb::sequence seq = nb::borrow<nb::sequence>(src);                             \
            if (nb::len(seq) != 3)                                                        \
                return false;                                                             \
            value.x = nb::cast<ElemType>(seq[0]);                                         \
            value.y = nb::cast<ElemType>(seq[1]);                                         \
            value.z = nb::cast<ElemType>(seq[2]);                                         \
            return true;                                                                  \
        }                                                                                 \
                                                                                          \
        static nb::handle from_cpp(const VecType &v, nb::rv_policy policy,                \
                                   nb::detail::cleanup_list *cleanup) noexcept {          \
            return nb::make_tuple(v.x, v.y, v.z).release();                               \
        }                                                                                 \
    };

// Macro for 4-component vector types
#define NB_TYPE_CASTER_VEC4(VecType, ElemType, type_name)                                 \
    template <>                                                                           \
    struct nb::detail::type_caster<VecType> {                                             \
        NB_TYPE_CASTER(VecType, const_name(type_name));                                   \
                                                                                          \
        bool from_python(nb::handle src, uint8_t flags, cleanup_list *cleanup) noexcept { \
            if (!nb::isinstance<nb::sequence>(src))                                       \
                return false;                                                             \
            nb::sequence seq = nb::borrow<nb::sequence>(src);                             \
            if (nb::len(seq) != 4)                                                        \
                return false;                                                             \
            value.x = nb::cast<ElemType>(seq[0]);                                         \
            value.y = nb::cast<ElemType>(seq[1]);                                         \
            value.z = nb::cast<ElemType>(seq[2]);                                         \
            value.w = nb::cast<ElemType>(seq[3]);                                         \
            return true;                                                                  \
        }                                                                                 \
                                                                                          \
        static nb::handle from_cpp(const VecType &v, nb::rv_policy policy,                \
                                   nb::detail::cleanup_list *cleanup) noexcept {          \
            return nb::make_tuple(v.x, v.y, v.z, v.w).release();                          \
        }                                                                                 \
    };

// Float types
NB_TYPE_CASTER_VEC2(float2, float, "float2")
NB_TYPE_CASTER_VEC3(float3, float, "float3")
NB_TYPE_CASTER_VEC4(float4, float, "float4")

// Int types
NB_TYPE_CASTER_VEC2(int2, int, "int2")
NB_TYPE_CASTER_VEC3(int3, int, "int3")
NB_TYPE_CASTER_VEC4(int4, int, "int4")

// Unsigned int types
NB_TYPE_CASTER_VEC2(uint2, unsigned int, "uint2")
NB_TYPE_CASTER_VEC3(uint3, unsigned int, "uint3")
NB_TYPE_CASTER_VEC4(uint4, unsigned int, "uint4")

#endif

// ================================================================================================================================

// template <> struct nb::detail::type_caster<float2> {
//     using Value = float2;
//     static constexpr auto Name = const_name("float2");
//     template <typename T_>
//     using Cast = movable_cast_t<T_>;
//     template <typename T_>
//     static constexpr bool can_cast() { return true; }
//     template <typename T_, enable_if_t<std::is_same_v<std::remove_cv_t<T_>, Value>> = 0>
//     static handle from_cpp(T_ *p, rv_policy policy, cleanup_list *list) {
//         if (!p) return none().release();
//         return from_cpp(*p, policy, list);
//     }
//     explicit operator Value *() { return &value; }
//     explicit operator Value &() { return (Value &)value; }
//     explicit operator Value &&() { return (Value &&)value; }
//     Value value;
//     ;
//     bool from_python(nb::handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
//         if (!nb::isinstance<nb::sequence>(src)) return false;
//         nb::sequence seq = nb::borrow<nb::sequence>(src);
//         if (nb::len(seq) != 2) return false;
//         value.x = nb::cast<float>(seq[0]);
//         value.y = nb::cast<float>(seq[1]);
//         return true;
//     }
//     static nb::handle from_cpp(const float2 &v, nb::rv_policy policy, nb::detail::cleanup_list *cleanup) noexcept { return nb::make_tuple(v.x, v.y).release(); }
// };

#undef CODE_ROUTE