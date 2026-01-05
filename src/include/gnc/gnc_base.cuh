/*

dynamic properties base template using CRTP and constexpr for automatic binding

*/
#pragma once
#include "template_macro_undef.h" // guard from defines

#include <array> // std::array (if you use arrays instead of tuples)
#include <optional>
#include <tuple>       // std::tuple, std::make_tuple
#include <type_traits> // optional but often useful for traits
#include <utility>     // std::forward, std::index_sequence, etc.

#include "core.h"
#include "core/cuda/types.cuh"
#include "macros.h"

// ================================================================================================================================

namespace gnc {

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

    int test123 = 123;

    // return properties plus defaults
    static constexpr auto properties() {
        return std::tuple_cat(Derived::properties_impl(),
                              std::make_tuple(
                                  // ================================================================
                                  // [Default Properties]
                                  // ----------------------------------------------------------------
                                  Property<Self, &Self::stream>{"stream", &Self::stream},
                                  Property<Self, &Self::test123>{"test123", &Self::test123}
                                  // ================================================================
                                  ));
    }

    virtual void process() = 0;
};

// ================================================================================================================================

// shortcuts used by the macros (standardized names)
using DeviceArrayFloat1D = core::Ref<core::cuda::DeviceArray<float, 1>>;
using DeviceArrayFloat2D = core::Ref<core::cuda::DeviceArray<float, 2>>;
using DeviceArrayFloat3D = core::Ref<core::cuda::DeviceArray<float, 3>>;

using DeviceArrayInt1D = core::Ref<core::cuda::DeviceArray<int, 1>>;
using DeviceArrayInt2D = core::Ref<core::cuda::DeviceArray<int, 2>>;
using DeviceArrayInt3D = core::Ref<core::cuda::DeviceArray<int, 3>>;

// ================================================================================================================================

} // namespace gnc