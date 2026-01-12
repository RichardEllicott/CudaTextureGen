/*

dynamic properties base template using CRTP and constexpr for automatic binding

*/
#pragma once
// #include "template_macro_undef.h" // guard from defines

#include <array> // std::array (if you use arrays instead of tuples)
#include <optional>
#include <tuple>       // std::tuple, std::make_tuple
#include <type_traits> // optional but often useful for traits
#include <utility>     // std::forward, std::index_sequence, etc.  std::swap

#include "core/cuda/device_struct.cuh"
#include "core/cuda/stream.cuh"
#include "core/cuda/types.cuh"
#include "macros.h"




#pragma region TRYING_TO_SLOVE_FLOAT2_EXPORT

#include "core/cuda/math.cuh"
#include "core/cuda/operators.cuh"



#pragma endregion

//
//





// ================================================================================================================================
// REFACTOR OPTIONS
// --------------------------------------------------------------------------------------------------------------------------------
#define REFACTOR_GNC_STORAGE_IN_PARS 0 // 0 is normal, 1 i was trying to refactor to point props to a member structure (lowering copying)
#define REFACTOR_GNC_TEMPLATE_VALIDATION
// #define ATTEMPT_GENERIC_REFLECTION // broken
// ================================================================================================================================

namespace gnc {

using namespace core::cuda::types; // include type aliases at top level

// ================================================================================================================================
// Template Validators
// --------------------------------------------------------------------------------------------------------------------------------

// alternate shorter, more generic??
template <typename T>
struct is_array_ref : std::false_type {};

template <typename T, int N>
struct is_array_ref<core::Ref<core::cuda::DeviceArray<T, N>>> : std::true_type {};

// static_assert(is_array_ref<T>::value, "T must be an array ref");

// --------------------------------------------------------------------------------------------------------------------------------

//
// extra is a ref?
template <typename T>
struct is_ref : std::false_type {};

template <typename U>
struct is_ref<core::Ref<U>> : std::true_type {};

// ================================================================================================================================

#ifdef ATTEMPT_GENERIC_REFLECTION

template <typename Tuple, typename F, std::size_t... I>
constexpr void tuple_for_each_impl(Tuple &&t, F &&f, std::index_sequence<I...>) {
    (f(std::get<I>(t)), ...);
}

template <typename Tuple, typename F>
constexpr void tuple_for_each(Tuple &&t, F &&f) {
    tuple_for_each_impl(
        std::forward<Tuple>(t),
        std::forward<F>(f),
        std::make_index_sequence<std::tuple_size_v<std::decay_t<Tuple>>>{});
}
#endif

//
//
//
//
// struct for properties
template <typename T, typename Member>
struct _Property {
    const char *name;
    Member member;
};

// helper to lower boilerplate in final form (see gnc_example for usage)
template <typename T, auto Member>
using Property = _Property<T, decltype(Member)>;

//
//
//
// 🧪 trying same for methods
template <typename T, auto MethodPtr>
struct Method {
    const char *name;
    static constexpr auto member = MethodPtr;
    // if your property metadata exposes __this, add it too:
    // using This = T;
    // static constexpr auto __this = static_cast<T*>(nullptr);
};

//
//
//

#if REFACTOR_GNC_STORAGE_IN_PARS == 1
// 🧪 attempting to bind properties direct to structure, is more complicated
template <typename T, auto SubobjectPtr, auto FieldPtr>
struct NestedProperty {
    const char *name;

    // Accessor
    auto &get(T &obj) const {
        return (obj.*SubobjectPtr).*FieldPtr;
    }
};
#endif

//
//
//

struct NoParams {}; // default if we don't use the params

template <typename Derived, typename Parameters = NoParams, typename ArrayPointers = NoParams>
class GNC_Base {
    using Self = GNC_Base;
    // ================================================================================================================================
  protected:
    // canonical CRTP
    Derived &derived() {
        return static_cast<Derived &>(*this);
    }
    // canonical CRTP
    const Derived &derived() const {
        return static_cast<const Derived &>(*this);
    }
    // ================================================================================================================================
  protected:
    Parameters pars;
    ArrayPointers array_pointers;

    core::cuda::DeviceStruct<Parameters> dev_pars;        // uploads parameters struct to device
    core::cuda::DeviceStruct<ArrayPointers> dev_array_pointers; // uploads array_pointers struct to device
    bool _pars_synced = false;

  public:
    core::Ref<core::cuda::Stream> stream; // gets a stream

    // ================================================================================================================================
    // [Set Par]
    // also marking the sync as dirty
    // --------------------------------------------------------------------------------------------------------------------------------
    // template for setting a par, is bound to python, will mark the _pars_synced
    template <typename T>
    void set_par(T &field, const T &value) {
        if (field != value) {
            // printf("value changed!\n");
            _pars_synced = false;
            field = value;
        }
    }
    // --------------------------------------------------------------------------------------------------------------------------------
    // opposite for clarity
    template <typename T>
    const T &get_par(const T &field) const {
        return field;
    }
    // ================================================================================================================================
    // [Ensure Array Ref Ready]
    // ensure the array ref is not empty, and if creating a new one set it up also marking sync as dirty
    // (ensuring pointers are uploaded before kernel launch)
    // --------------------------------------------------------------------------------------------------------------------------------

    // template <typename Container, size_t Dim>
    // std::array<size_t, Dim> to_size_array(const Container &c) {
    //     std::array<size_t, Dim> out{};
    //     for (size_t i = 0; i < Dim; ++i)
    //         out[i] = static_cast<size_t>(c[i]);
    //     return out;
    // }

    // template <
    //     typename MapT,
    //     typename ShapeT,
    //     typename = std::enable_if_t<gnc::is_array_ref<MapT>::value>>
    // inline void ensure_array_ref_ready(
    //     MapT &map, const ShapeT &desired_shape, bool zero_device = false) {

    //     map.instantiate_if_null();

    //     auto desired = to_size_array(desired_shape); // canonicalize here

    //     if (map->shape() != desired) {
    //         _pars_synced = false;
    //         map->resize(desired);
    //         if (zero_device) map->zero_device();
    //     }
    // }

    template <
        typename MapT,
        typename ShapeT,
        typename = std::enable_if_t<gnc::is_array_ref<MapT>::value>>
    inline void ensure_array_ref_ready(
        MapT &map, const ShapeT &desired_shape, bool zero_device = false) {

        map.instantiate_if_null(); // ensure the Ref is not empty

        if (map->shape() != desired_shape) { // if shape missmatch we will resize
            _pars_synced = false;      // and mark sync dirty (to trigger par upload)
            map->resize(desired_shape);
            if (zero_device) map->zero_device();
        }
    }

    // ================================================================================================================================

    // ready device ensuring par structs are uploaded
    void ready_device() {

        if (_pars_synced) return; // skip if already synced
        stream.instantiate_if_null();   // ensure we have a stream

        derived()._ready_device();

        stream->sync(); // wait on stream, to ensure copying completes
        _pars_synced = true;
    }

    // ================================================================================================================================
    // [constexpr Reflection]
    // --------------------------------------------------------------------------------------------------------------------------------
    // return properties plus defaults
    static constexpr auto properties() {
        return std::tuple_cat(Derived::_properties(), // CRTP requirement
                              std::tuple{
                                  // ================================================================
                                  // [Default Properties]
                                  // ----------------------------------------------------------------
                                  Property<Self, &Self::stream>{"stream", &Self::stream},
                                  //   Property<Self, &Self::width>{"width", &Self::width},
                                  //   Property<Self, &Self::height>{"width", &Self::height},
                                  // ================================================================
                              });
    }

    // return properties plus defaults
    static constexpr auto properties2() {
        return std::tuple_cat(Derived::_properties2(), // CRTP requirement
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

    // 🧪 trying same with methods
    // return properties plus defaults
    static constexpr auto methods() {
        return std::tuple_cat(Derived::_methods(), // CRTP requirement
                              std::tuple{

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

// (NAME)
#define REF_DEVICE_ARRAY_TYPES \
    X(RefDeviceArrayFloat1D)   \
    X(RefDeviceArrayFloat2D)   \
    X(RefDeviceArrayFloat3D)   \
    X(RefDeviceArrayInt1D)     \
    X(RefDeviceArrayInt2D)     \
    X(RefDeviceArrayInt3D)
#ifdef REF_DEVICE_ARRAY_TYPES
#define X(NAME)                               \
    for (auto *arr : get_all_of_type<NAME>()) \
        arr->instantiate_if_null();
        REF_DEVICE_ARRAY_TYPES
#undef X
#endif
#undef REF_DEVICE_ARRAY_TYPES
    }

#ifdef ATTEMPT_GENERIC_REFLECTION // broken on linux

    void instantiate_all_arrays2() {

        auto props = Self::properties();

        tuple_for_each(Self::properties(), [&](auto &prop) {
            using PropT = decltype(prop);
            using MemberT = typename PropT::member_type;

            if constexpr (gnc::is_ref<MemberT>::value) {
                auto &ref = this->*(PropT::member_ptr);
                // handle Ref<T>
            } else {
                auto &value = this->*(PropT::member_ptr);
                // handle normal property
            }
        });
    }
#endif

    GNC_Base() {
        // instantiate_all_arrays();
        stream.instantiate_if_null();
    }

    // main execution function, ensure device ready and run
    void compute() {
        // ready_device();      // ensure device ready
        derived()._compute();
    }
    //
};

} // namespace gnc

#undef DEVICE_ARRAY_TYPES