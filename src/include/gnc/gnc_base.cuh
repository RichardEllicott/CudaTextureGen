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

// #define ADD_REFLECTION_PATTERN_1 // optional reflection pattern, might be best to do this in python
// #define ADD_REFLECTION_PATTERN_2 // optional reflection pattern, might be best to do this in python
#define ADD_REFLECTION_PATTERN_RUNTIME_1 // optional reflection pattern, might be best to do this in python

namespace gnc {

// ================================================================================================================================
// standard DeviceArray Refs
using DeviceArrayFloat1D = core::Ref<core::cuda::DeviceArray<float, 1>>;
using DeviceArrayFloat2D = core::Ref<core::cuda::DeviceArray<float, 2>>;
using DeviceArrayFloat3D = core::Ref<core::cuda::DeviceArray<float, 3>>;

using DeviceArrayInt1D = core::Ref<core::cuda::DeviceArray<int, 1>>;
using DeviceArrayInt2D = core::Ref<core::cuda::DeviceArray<int, 2>>;
using DeviceArrayInt3D = core::Ref<core::cuda::DeviceArray<int, 3>>;

// ================================================================================================================================
#ifdef ADD_REFLECTION_PATTERN_1

// Trait to detect Ref<T>
template <typename T>
struct is_ref : std::false_type {};

template <typename U>
struct is_ref<core::Ref<U>> : std::true_type {};

// Helper: instantiate Ref<T> if the property is one
template <typename Self, typename Prop>
static void instantiate_if_ref(Self &self, const Prop &prop) {
    // Extract the actual member type, e.g. core::Ref<Texture>
    using MemberT = decltype(self.*(prop.member));

    if constexpr (is_ref<MemberT>::value) {
        // Call instantiate() on the Ref<T>
        (self.*(prop.member)).instantiate();
    }
}

#endif
// ================================================================================================================================
#ifdef ADD_REFLECTION_PATTERN_2

// extract the actual member type from a property
template <typename Self, typename Prop>
using member_type_t = decltype(std::declval<Self>().*(Prop::member));

// Filter a tuple of properties by member type
template <typename T, typename Self, typename Tuple, std::size_t... I>
auto filter_properties_impl(Self &self, Tuple &tup, std::index_sequence<I...>) {
    return std::tuple{
        (
            std::conditional_t<
                std::is_same_v<member_type_t<Self, std::tuple_element_t<I, Tuple>>, T>,
                std::tuple<member_type_t<Self, std::tuple_element_t<I, Tuple>> *>,
                std::tuple<>>{
                std::is_same_v<member_type_t<Self, std::tuple_element_t<I, Tuple>>, T>
                    ? &(self.*(std::get<I>(tup).member))
                    : nullptr})...};
}
#endif
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

    int test123 = 123;

    // return properties plus defaults
    static constexpr auto properties() {
        return std::tuple_cat(Derived::properties_impl(),
                              std::tuple{
                                  // ================================================================
                                  // [Default Properties]
                                  // ----------------------------------------------------------------
                                  Property<Self, &Self::stream>{"stream", &Self::stream},
                                  Property<Self, &Self::test123>{"test123", &Self::test123}
                                  // ================================================================
                              });
    }

// ================================================================================================================================
#ifdef ADD_REFLECTION_PATTERN_1
    // instantiate all Ref<T> properties automatically
    void instantiate_refs() {
        auto &self = static_cast<Derived &>(*this);

        std::apply(
            [&](auto &...props) {
                (instantiate_if_ref(self, props), ...);
            },
            Derived::properties());
    }
#endif
// ================================================================================================================================
#ifdef ADD_REFLECTION_PATTERN_2
    // ------------------------------------------------------------
    // NEW: return all members of type T as a tuple of pointers
    // ------------------------------------------------------------
    template <typename T>
    auto get_all_of_type() {
        auto &self = static_cast<Derived &>(*this);
        auto props = Derived::properties();

        return filter_properties_impl<T>(
            self,
            props,
            std::make_index_sequence<std::tuple_size_v<decltype(props)>>{});
    }
#endif
    // ================================================================================================================================
#ifdef ADD_REFLECTION_PATTERN_RUNTIME_1

    // return vector pointers to members whose type is exactly T
    template <typename T>
    auto get_all_of_type() {
        auto &self = static_cast<Derived &>(*this);
        std::vector<T *> result;

        std::apply(
            [&](auto &...props) {
                (([&] {
                     using MemberT = decltype(self.*(props.member));
                     if (typeid(MemberT) == typeid(T)) {
                         result.push_back(&(self.*(props.member)));
                     }
                 }()),
                 ...);
            },
            Derived::properties());

        return result;
    }

#endif
    // ================================================================================================================================

    void instantiate_all_arrays() {
        for (auto *arr : get_all_of_type<DeviceArrayFloat1D>()) arr->instantiate();
        for (auto *arr : get_all_of_type<DeviceArrayFloat2D>()) arr->instantiate();
        for (auto *arr : get_all_of_type<DeviceArrayFloat3D>()) arr->instantiate();
        for (auto *arr : get_all_of_type<DeviceArrayInt1D>()) arr->instantiate();
        for (auto *arr : get_all_of_type<DeviceArrayInt2D>()) arr->instantiate();
        for (auto *arr : get_all_of_type<DeviceArrayInt3D>()) arr->instantiate();
    }

    GNC_Base() {
    }

    virtual void process() = 0;
};

} // namespace gnc