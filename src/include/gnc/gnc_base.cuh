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

// ================================================================================================================================
// REFACTOR OPTIONS
// --------------------------------------------------------------------------------------------------------------------------------
#define REFACTOR_GNC_STORAGE_IN_PARS 0 // 0 is normal, 1 i was trying to refactor to point props to a member structure (lowering copying)

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
    Parameters parameters;
    ArrayPointers array_pointers;

    core::cuda::DeviceStruct<Parameters> dev_parameters;        // uploads parameters struct to device
    core::cuda::DeviceStruct<ArrayPointers> dev_array_pointers; // uploads array_pointers struct to device
    bool _parameters_synced = false;

  public:
    core::Ref<core::cuda::Stream> stream; // gets a stream

    // ================================================================================================================================
    // [Set Par]
    // also marking the sync as dirty
    // --------------------------------------------------------------------------------------------------------------------------------
    // template for setting a par, is bound to python, will mark the _parameters_synced
    template <typename T>
    void set_par(T &field, const T &value) {
        if (field != value) {
            // printf("value changed!\n");
            _parameters_synced = false;
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

#define CODE_ROUTE 1 // route 0 has restrictions but is failing linux?
#if CODE_ROUTE == 0

    template <typename T>
    struct is_array_ref : std::false_type {};
    template <>
    struct is_array_ref<core::Ref<core::cuda::DeviceArray<int, 1>>> : std::true_type {};
    template <>
    struct is_array_ref<core::Ref<core::cuda::DeviceArray<int, 2>>> : std::true_type {};
    template <>
    struct is_array_ref<core::Ref<core::cuda::DeviceArray<int, 3>>> : std::true_type {};
    template <>
    struct is_array_ref<core::Ref<core::cuda::DeviceArray<float, 1>>> : std::true_type {};
    template <>
    struct is_array_ref<core::Ref<core::cuda::DeviceArray<float, 2>>> : std::true_type {};
    template <>
    struct is_array_ref<core::Ref<core::cuda::DeviceArray<float, 3>>> : std::true_type {};

    // --------------------------------------------------------------------------------------------------------------------------------
    template <typename MapT, typename ShapeT>
    inline std::enable_if_t<is_array_ref<MapT>::value>
    ensure_array_ref_ready(MapT &map, const ShapeT &desired_shape, bool zero_device = false) {
        map.instantiate_if_null();

        if (map->shape() != desired_shape) {
            _parameters_synced = false; // we will need a new pointer upload
            map->resize(desired_shape);
            if (zero_device)
                map->zero_device();
        }
    }

#elif CODE_ROUTE == 1

    template <typename MapT, typename ShapeT>
    void  ensure_array_ref_ready(MapT &map, const ShapeT &desired_shape, bool zero_device = false) {
        map.instantiate_if_null();

        if (map->shape() != desired_shape) {
            _parameters_synced = false; // we will need a new pointer upload
            map->resize(desired_shape);
            if (zero_device)
                map->zero_device();
        }
    }

#endif
#undef CODE_ROUTE





    // ================================================================================================================================

    // ready device ensuring par structs are uploaded
    void ready_device() {

        if (_parameters_synced) return; // skip if already synced
        stream.instantiate_if_null();   // ensure we have a stream

        derived()._ready_device();

        stream->sync(); // wait on stream, to ensure copying completes
        _parameters_synced = true;
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