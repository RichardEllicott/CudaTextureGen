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

    // [REFLECTION ]
    // return vector pointers to members whose type is exactly T
    template <typename T>
    auto get_all_of_type() {
        auto &self = static_cast<Derived &>(*this);
        std::vector<T *> result;

        std::apply(
            [&](auto &...props) {
                (([&] {
                     using MemberT = decltype(self.*(props.member));

                     if constexpr (std::is_same_v<MemberT, T>) {
                         result.push_back(&(self.*(props.member)));
                     }
                 }()),
                 ...);
            },
            Derived::properties());

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