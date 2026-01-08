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

#include "core/cuda/device_struct.cuh"
#include "core/cuda/stream.cuh"
#include "core/cuda/types.cuh"
#include "macros.h"

namespace gnc {

using namespace core::cuda::types; // include type aliases at top level

// struct for properties
template <typename T, typename Member>
struct _Property {
    const char *name;
    Member member;
};

// helper to lower boilerplate in final form (see gnc_example for usage)
template <typename T, auto Member>
using Property = _Property<T, decltype(Member)>;

struct NoParams {}; // default if we don't use the params

template <typename Derived, typename Parameters = NoParams, typename ArrayPointers = NoParams>
class GNC_Base {
    using Self = GNC_Base;

  protected:
    core::cuda::DeviceStruct<Parameters> parameters;        // uploads parameters struct to device
    core::cuda::DeviceStruct<ArrayPointers> array_pointers; // uploads array_pointers struct to device

    bool _parameters_synced = false;

  public:
    // template for setting a par, this will mark the device as requiring a new parameters upload
    // this template must be called specially by the python bindings
    template <typename T>
    void set_par(T &field, const T &value) {
        if (field != value) {
            printf("value changed!\n");
            _parameters_synced = false;
            field = value;
        }
    }

    core::Ref<core::cuda::Stream> stream; // gets a stream

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

    // return properties plus defaults
    static constexpr auto properties2() {
        return std::tuple_cat(Derived::properties2_impl(),
                              std::tuple{
                                  // ================================================================
                                  // [Default Properties]
                                  // ----------------------------------------------------------------
                                  //   Property<Self, &Self::stream>{"stream", &Self::stream},
                                  //   Property<Self, &Self::width>{"width", &Self::width},
                                  //   Property<Self, &Self::height>{"width", &Self::height},
                                  // ================================================================
                              });
    }

    // ================================================================================================================================
    // [OPTIONAL REFLECTION] return vector pointers to members whose type is exactly T
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