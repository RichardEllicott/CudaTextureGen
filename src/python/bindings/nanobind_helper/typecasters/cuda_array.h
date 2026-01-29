/*

typecasters for the following:

core::cuda::types::Array
    (reproduction of std::Array but cuda compatible)

core::cuda::types::BoolArray
    (reproduction of std::Array but cuda compatible and uses ints device side, bools python side)


*/
#pragma once

#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>

#include "core/cuda/types.cuh"

namespace nb = nanobind;



template <typename T, size_t N>
struct nb::detail::type_caster<core::cuda::types::Array<T, N>> {
    using Value = core::cuda::types::Array<T, N>;
    static constexpr auto Name = const_name("Array");
    
    template <typename T_>
    using Cast = movable_cast_t<T_>;
    
    template <typename T_>
    static constexpr bool can_cast() { return true; }
    
    template <typename T_, enable_if_t<std::is_same_v<std::remove_cv_t<T_>, Value>> = 0>
    static handle from_cpp(T_ *p, rv_policy policy, cleanup_list *list) {
        if (!p) return none().release();
        return from_cpp(*p, policy, list);
    }
    
    explicit operator Value *() { return &value; }
    explicit operator Value &() { return (Value &)value; }
    explicit operator Value &&() { return (Value &&)value; }
    
    Value value;
    
    bool from_python(nb::handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
        if (!nb::isinstance<nb::sequence>(src)) 
            return false;
        
        nb::sequence seq = nb::borrow<nb::sequence>(src);
        if (nb::len(seq) != N) 
            return false;
        
        for (size_t i = 0; i < N; ++i) {
            value[i] = nb::cast<T>(seq[i]);
        }
        
        return true;
    }
    
    static nb::handle from_cpp(const Value &v, nb::rv_policy policy, nb::detail::cleanup_list *cleanup) noexcept {
        nb::list out;
        for (size_t i = 0; i < N; ++i) {
            out.append(nb::cast(v[i]));
        }
        return out.release();
    }
};

// ================================================================================================================================



template <size_t N>
struct nb::detail::type_caster<core::cuda::types::BoolArray<N>> {
    using Value = core::cuda::types::BoolArray<N>;
    static constexpr auto Name = const_name("BoolArray");
    
    template <typename T_>
    using Cast = movable_cast_t<T_>;
    
    template <typename T_>
    static constexpr bool can_cast() { return true; }
    
    template <typename T_, enable_if_t<std::is_same_v<std::remove_cv_t<T_>, Value>> = 0>
    static handle from_cpp(T_ *p, rv_policy policy, cleanup_list *list) {
        if (!p) return none().release();
        return from_cpp(*p, policy, list);
    }
    
    explicit operator Value *() { return &value; }
    explicit operator Value &() { return (Value &)value; }
    explicit operator Value &&() { return (Value &&)value; }
    
    Value value;
    
    bool from_python(nb::handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
        if (!nb::isinstance<nb::sequence>(src)) 
            return false;
        
        nb::sequence seq = nb::borrow<nb::sequence>(src);
        if (nb::len(seq) != N) 
            return false;
        
        for (size_t i = 0; i < N; ++i) {
            value.set(i, nb::cast<bool>(seq[i]));
        }
        
        return true;
    }
    
    static nb::handle from_cpp(const Value &v, nb::rv_policy policy, nb::detail::cleanup_list *cleanup) noexcept {
        nb::list out;
        for (size_t i = 0; i < N; ++i) {
            out.append(nb::cast(v[i]));
        }
        return out.release();
    }
};
// ================================================================================================================================
