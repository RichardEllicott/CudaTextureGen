/*

dynamic properties base template using CRTP and constexpr for automatic binding

*/
#pragma once
// #include "template_macro_undef.h" // guard from defines

#include <array> // std::array (if you use arrays instead of tuples)
#include <optional>
#include <tuple>       // std::tuple, std::make_tuple
#include <type_traits> // optional but often useful for traits
#include <utility>     // std::forward, std::index_sequence, etc.

#include "core.h"
#include "core/cuda/types.cuh"
#include "macros.h"

namespace gnc {

// ================================================================================================================================
// [⚠️ TODO will add these to core/cuda/types.cuh]
// --------------------------------------------------------------------------------------------------------------------------------
// standard DeviceArray Refs
using DeviceArrayFloat1D = core::Ref<core::cuda::DeviceArray<float, 1>>;
using DeviceArrayFloat2D = core::Ref<core::cuda::DeviceArray<float, 2>>;
using DeviceArrayFloat3D = core::Ref<core::cuda::DeviceArray<float, 3>>;

using DeviceArrayInt1D = core::Ref<core::cuda::DeviceArray<int, 1>>;
using DeviceArrayInt2D = core::Ref<core::cuda::DeviceArray<int, 2>>;
using DeviceArrayInt3D = core::Ref<core::cuda::DeviceArray<int, 3>>;

// (NAME)
#define DEVICE_ARRAY_TYPES \
    X(DeviceArrayFloat1D)  \
    X(DeviceArrayFloat2D)  \
    X(DeviceArrayFloat3D)  \
    X(DeviceArrayInt1D)    \
    X(DeviceArrayInt2D)    \
    X(DeviceArrayInt3D)

// std::array aliases
using Float2 = std::array<float, 2>;
using Float3 = std::array<float, 3>;
using Float4 = std::array<float, 4>;
using Float5 = std::array<float, 5>;
using Float6 = std::array<float, 6>;
using Float7 = std::array<float, 7>;
using Float8 = std::array<float, 8>;

using Int2 = std::array<int, 2>;
using Int3 = std::array<int, 3>;
using Int4 = std::array<int, 4>;
using Int5 = std::array<int, 5>;
using Int6 = std::array<int, 6>;
using Int7 = std::array<int, 7>;
using Int8 = std::array<int, 8>;

// host side helpers to convert to cuda versions

// Float2 => float2
inline float2 to_float2(const Float2 &val) { return {val[0], val[1]}; }
// Float3 => float3
inline float3 to_float3(const Float3 &val) { return {val[0], val[1], val[2]}; }
// Float4 => float4
inline float4 to_float4(const Float4 &val) { return {val[0], val[1], val[2], val[3]}; }

// Int2 => int2
inline int2 to_int2(const Int2 &val) { return {val[0], val[1]}; }
// Int3 => int3
inline int3 to_int3(const Int3 &val) { return {val[0], val[1], val[2]}; }
// Int4 => int4
inline int4 to_int4(const Int4 &val) { return {val[0], val[1], val[2], val[3]}; }

// ================================================================================================================================

// struct for properties
template <typename T, typename Member>
struct _Property {
    const char *name;
    Member member;
};

// helper to lower boilerplate in final form (see gnc_example for usage)
template <typename T, auto Member>
using Property = _Property<T, decltype(Member)>;

template <typename Derived>
class GNC_Base {
    using Self = GNC_Base;

  public:
    core::Ref<core::cuda::Stream> stream;

    int width = 128;
    int height = 128;
    // dim3 block(16, 16); // ⚠️ breaks cuda

    // return properties plus defaults
    static constexpr auto properties() {
        return std::tuple_cat(Derived::properties_impl(),
                              std::tuple{
                                  // ================================================================
                                  // [Default Properties]
                                  // ----------------------------------------------------------------
                                  Property<Self, &Self::stream>{"stream", &Self::stream},
                                  Property<Self, &Self::width>{"width", &Self::width},
                                  Property<Self, &Self::height>{"width", &Self::height},
                                  // ================================================================
                              });
    }

    // ================================================================================================================================

    // previous version worked with MSVC only
    // [REFLECTION ]
    // return vector pointers to members whose type is exactly T
    // template <typename T>
    // auto get_all_of_type() {
    //     auto &self = static_cast<Derived &>(*this);
    //     std::vector<T *> result;

    //     std::apply(
    //         [&](auto &...props) {
    //             (([&] {
    //                  using MemberT = decltype(self.*(props.member));

    //                  if constexpr (std::is_same_v<MemberT, T>) {
    //                      result.push_back(&(self.*(props.member)));
    //                  }
    //              }()),
    //              ...);
    //         },
    //         Derived::properties());

    //     return result;
    // }
    //
    //
    //

    // corrected
    template <typename T>
    auto get_all_of_type() {
        // auto &self = static_cast<Derived &>(*this); // GCC didn't like this
        Derived &self = static_cast<Derived &>(*this); //

        std::vector<T *> result;

        auto props = Derived::properties();

        std::apply(
            [&](auto &...prop) {
                // process each property individually
                ([&](auto &p) {
                    using MemberT = decltype(self.*(p.member));
                    using DecayedMemberT = std::remove_reference_t<MemberT>;

                    if constexpr (std::is_same_v<DecayedMemberT, T>) {
                        result.push_back(&(self.*(p.member)));
                    }
                }(prop),
                 ...);
            },
            props);

        return result;
    }

    // ================================================================================================================================

    // using get_all_of_type, ensure all arrays are instantiated (they will still be empty though)
    void instantiate_all_arrays() {
#ifdef DEVICE_ARRAY_TYPES
#define X(NAME)                               \
    for (auto *arr : get_all_of_type<NAME>()) \
        arr->instantiate_if_null();
        DEVICE_ARRAY_TYPES
#undef X
#endif
    }

    GNC_Base() {
        instantiate_all_arrays();
        stream.instantiate_if_null();
    }

    virtual void process() = 0; // must be implemented by child
};

} // namespace gnc

#undef DEVICE_ARRAY_TYPES