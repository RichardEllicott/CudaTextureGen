/*

dynamic properties base template using CRTP and constexpr for automatic binding

*/
#pragma once
// #include "template_macro_undef.h" // guard from defines

// #include <array> // std::array
#include <cassert>
#include <functional> // std::invoke
#include <optional>
#include <stdexcept>   // exceptions
#include <tuple>       // std::tuple, std::make_tuple
#include <type_traits> // optional but often useful for traits
#include <utility>     // std::forward, std::index_sequence, etc.  std::swap

#include "core/cuda/cast.cuh" // top level for this object
#include "core/cuda/device_struct.cuh"
#include "core/cuda/hash.cuh"
#include "core/cuda/math.cuh"
#include "core/cuda/stream.cuh"
#include "core/cuda/types.cuh" // top level for this object
#include "macros.h"

#include "core/reflection.h"

#include "core/util.h"

#define GNC_BASE_LAZY_EVALUATION

namespace gnc {

// ================================================================================================================================

using namespace core::cuda::types; // include type aliases at top level
using namespace core::cuda::cast;  // include type aliases at top level
using namespace core::reflection;  // include at top level

namespace cmath = core::cuda::math; // include the cuda math lib as cmath
namespace chash = core::cuda::hash; // include the cuda math lib as cmath

#pragma region VALIDATORS // check if something is a particular type by template

// ================================================================================================================================
// [core::Ref<core::cuda::DeviceArray<T, N>>]
template <typename T>
struct is_device_array_ref : std::false_type {};
template <typename T, int N>
struct is_device_array_ref<core::Ref<core::cuda::DeviceArray<T, N>>> : std::true_type {};
// --------------------------------------------------------------------------------------------------------------------------------
// [core::Ref<U>]
template <typename T>
struct is_ref : std::false_type {};
template <typename U>
struct is_ref<core::Ref<U>> : std::true_type {};
// ================================================================================================================================

#pragma endregion

struct NoParams {}; // default if we don't use the params

template <typename Derived, typename Parameters = NoParams, typename ArrayPointers = NoParams>
class GNC_Base {

  protected:
    // CRTP: obtain this object as the concrete Derived type.
    Derived &derived() { return static_cast<Derived &>(*this); }
    const Derived &derived() const { return static_cast<const Derived &>(*this); }

  protected:
    // static core::reflection::Reflection<Derived> reflection; // using a generalize reflection pattern

    // ================================================================================================================================
  protected:
    Parameters _pars;
    ArrayPointers _arrays;

    core::cuda::DeviceStruct<Parameters> _dev_pars;      // uploads parameters struct to device
    core::cuda::DeviceStruct<ArrayPointers> _dev_arrays; // uploads array_pointers struct to device
    bool _pars_synced = false;

  public:
    core::Ref<core::cuda::Stream> stream; // gets a stream

    // ================================================================================================================================
    // [Set Par]
    // also marking the sync as dirty (bound by nanobind)
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
    // EXAMPLE:
    // ensure_array_ref_ready(water_map, height_map->shape(), true);
    // --------------------------------------------------------------------------------------------------------------------------------

    // simple helper to make ensuring an array exists with desired shape (if it doesn't already) quicker
    template <
        typename MapT,
        typename ShapeT,
        typename = std::enable_if_t<gnc::is_device_array_ref<MapT>::value>>
    inline void ensure_array_ref_ready(
        MapT &map, const ShapeT &desired_shape, bool zero_device = false) {

        map.instantiate_if_null(); // ensure the Ref is not empty

        if (map->shape() != desired_shape) { // if shape missmatch we will resize
            _pars_synced = false;            // and mark sync dirty (to trigger par upload)
            map->resize(desired_shape);
            if (zero_device) map->zero_device();
        }
    }

    // ================================================================================================================================

    // ================================================================================================================================
    // [constexpr Reflection]
    // --------------------------------------------------------------------------------------------------------------------------------
    // return properties

    // move to bottom

#pragma region COMPILE_TIME_REFLECTION // getting all of type, to do stuff like make all arrays valid etc

    // get pointer list to all of type T
    // note used for multiple types could add more compile time overhead
    template <typename T>
    std::vector<T *> ct_get_all_of_type() {

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

    // compile time reflection pattern that ensures all Ref's are instantiated
    void instantiate_all_refs() {
        // printf("instantiate_all_refs()...\n");

        constexpr auto props = Derived::properties();

        for_each_property(props, [&](auto const &p) {
            // printf("prop: %s\n", p.name);
            using MemberT = decltype(std::declval<Derived>().*(p.member));
            using RawMemberT = std::remove_cv_t<std::remove_reference_t<MemberT>>;

            // is_ref
            if constexpr (is_ref<RawMemberT>::value) {
                // printf("is_ref == true!\n");
                auto &ref = derived().*(p.member);
                ref.instantiate_if_null();
            }
            // // is_device_array_ref
            // if constexpr (is_device_array_ref<RawMemberT>::value) {
            //     // printf("is_device_array_ref == true!\n");
            //     // auto &value = derived().*(p.member); // should point to our member
            // }

            // // get actual device array
            // if constexpr (is_device_array_ref<RawMemberT>::value) {

            //     // Access the actual Ref<DeviceArray<T,N>> instance
            //     auto &ref = derived().*(p.member);

            //     // ----------------------------------------------------------------
            //     // ⚠️ breaks GCC
            //     // core::RefBase &ref_base = ref; // for autotype
            //     // ref_base.instantiate_if_null();
            //     // ⚠️ also breaks GCC
            //     // core::RefBase &ref_base = static_cast<core::RefBase &>(ref);
            //     // ref_base.instantiate_if_null();
            //     // ----------------------------------------------------------------

            //     // Safety check (optional)
            //     if (!ref) {
            //         printf("  %s is empty\n", p.name);
            //         return;
            //     }

            //     // Now get the device pointer
            //     auto *ptr = ref->dev_ptr();

            //     set_property_by_name(_arrays, p.name, ptr);

            //     printf("  %s -> dev_ptr() = %p\n", p.name, (void *)ptr);
            // }
        });
    }

#pragma endregion

#pragma region RUNTIME_REFLECTION

    // runtime property list (static lazy)
    static const std::vector<RuntimeProperty> &runtime_properties() {
        static const std::vector<RuntimeProperty> props =
            build_runtime_properties_from_tuple<Derived>(Derived::properties());
        return props;
    }

    // runtime property map (static lazy)
    static const auto &runtime_property_map() {
        static const auto map = core::util::to_unordered_map(runtime_properties()); // lazy
        return map;
    }

    // ================================================================================================================================

// refactoring to use a helper object, note might break lazy design intention!!
#define BASE_REFACTOR_TO_REFLECTION_OB 0 // trying to seperate the code off to Reflection object (didn't work)
#if BASE_REFACTOR_TO_REFLECTION_OB == 0

    // get all of type from runtime_properties()
    template <typename T>
    auto get_properties_of_type() {
        using CleanT = std::remove_cv_t<std::remove_reference_t<T>>;

        std::vector<T *> result;
        auto &self = derived();

        for (auto const &rp : runtime_properties()) {
            if (*rp.type == typeid(CleanT)) {
                auto *raw_ptr = reinterpret_cast<char *>(&self) + rp.offset;
                auto *ptr = reinterpret_cast<T *>(raw_ptr);
                result.push_back(ptr);
            }
        }

        return result;
    }

#elif BASE_REFACTOR_TO_REFLECTION_OB == 1 // trying to use a helper object

    template <typename T>
    auto get_properties_of_type() {
        auto &self = derived();
        return Reflection<Derived>::template get_properties_of_type<RefDeviceArrayFloat2D>(self);
    }

#endif
#undef BASE_REFACTOR_TO_REFLECTION_OB

#pragma endregion

#pragma region TESTS

    // compile time test
    void _instance_test_1() {
        printf("_instance_test_1()...\n");

        for (auto *arr : ct_get_all_of_type<RefDeviceArrayFloat2D>()) {
            printf(" arr->instantiate_if_null()...\n");
            arr->instantiate_if_null();
        }
    }

    void _instance_test_2() {
        printf("_instance_test_2()...\n");

        for (auto *arr : get_properties_of_type<RefDeviceArrayFloat2D>()) {
            printf(" arr->instantiate_if_null()...\n");
            arr->instantiate_if_null();
        }
    }

    int _return_int_test(int v) {
        return v;
    }

    void _init_tests() {
    }

#pragma endregion

#pragma region COPY_TO_ARRAY_WITH_RUNTIME_REFLECTION

#define PART_1
#define PART_2

    // copy the properties to structure
    // ⚠️ This reflection system requires trivially-copyable POD types due to memcpy
    void _copy_pars_and_array_ptrs_to_structs() {

        bool debug_print = false;

        if (debug_print) printf("_copy_pars_and_array_ptrs_to_structs()...\n");

        auto &_props_map = runtime_property_map();

#ifdef PART_1

        // list of par properties
        static const std::vector<RuntimeProperty> _pars_props =
            build_runtime_properties_from_tuple<Parameters>(Parameters::properties());

        // for each prop
        for (auto &par_prop : _pars_props) {

            // Destination: inside _pars
            void *dst_ptr = get_ptr_from_runtime_property(par_prop, _pars);

            if (auto it = _props_map.find(par_prop.name); it != _props_map.end()) {

                if (debug_print) printf("matched parameter '%s'\n", par_prop.name);

                const RuntimeProperty *inst_prop = &it->second;

                // check types match
                if (*par_prop.type != *inst_prop->type) {
                    printf("Type mismatch for '%s' — skipping copy\n", par_prop.name);
                    continue;
                }

                void *src_ptr = get_ptr_from_runtime_property(*inst_prop, derived());

                if (debug_print && *par_prop.type == typeid(int)) {
                    int value = *reinterpret_cast<int *>(dst_ptr);
                    printf("int value before copy = %d\n", value);
                }

                memcpy(dst_ptr, src_ptr, par_prop.size); // copy from source to destination

                // Debug: print the copied value if it's an int
                if (debug_print && *par_prop.type == typeid(int)) {
                    int value = *reinterpret_cast<int *>(dst_ptr);
                    printf("int value after copy = %d\n", value);
                }
            }
        }

#endif

#ifdef PART_2

        // list of array props
        static const std::vector<RuntimeProperty> _arrays_props =
            build_runtime_properties_from_tuple<ArrayPointers>(ArrayPointers::properties());

        for (auto &arrays_prop : _arrays_props) {

            void *dst_ptr = get_ptr_from_runtime_property(arrays_prop, _arrays);
            *reinterpret_cast<void **>(dst_ptr) = nullptr; // set dst_ptr to nullptr (default)

            if (auto it = _props_map.find(arrays_prop.name); it != _props_map.end()) {
                if (debug_print) printf("matched array parameter '%s'\n", arrays_prop.name);

                const RuntimeProperty *inst_prop = &it->second;

                void *src_ptr = get_ptr_from_runtime_property(*inst_prop, derived());

                auto *ref = reinterpret_cast<core::RefBase *>(src_ptr); // might be unsafe?

                assert(dynamic_cast<core::RefBase *>(ref) &&
                       "Reflection error: expected property to be a Ref<T> (invalid src_ptr type)"); // OPTIONAL check

                if (ref->is_valid()) {
                    void *obj_ptr = ref->raw_ptr();

                    // reinterpret_cast first (just changes type does not check)
                    // dynamic_cast checks, will give nullptr if not valid
                    auto *arr = dynamic_cast<core::cuda::DeviceArrayBase *>(
                        reinterpret_cast<core::cuda::DeviceArrayBase *>(obj_ptr));

                    assert(arr && "Reflection error: expected DeviceArrayBase"); // OPTIONAL check

                    if (arr) *reinterpret_cast<void **>(dst_ptr) = arr->raw_dev_ptr();
                }
            }
        }

#endif

#undef PART_1
#undef PART_2
    }

#pragma endregion

    // ready device ensuring par structs are uploaded
    void ready_device() {

        if (_pars_synced) return;     // skip if already synced
        stream.instantiate_if_null(); // ensure we have a stream

        _dev_pars.set_stream(stream->get()); // ensure streams
        _dev_arrays.set_stream(stream->get());

        derived()._ready_device(); // run derived which is set up by macro (currently copies all the vars to the pars by macro)

        _copy_pars_and_array_ptrs_to_structs(); // 🧪 new reflection pattern, might be a tad slower than macros on boilerplate

        _dev_pars.upload(_pars);     // upload pars
        _dev_arrays.upload(_arrays); // upload array pointers

        stream->sync();      // wait on stream, to ensure copying completes
        _pars_synced = true; // mark the pars as having synced, should prevent uploading unless pars have changed
    }

    GNC_Base() {
        _init_tests();
        // instantiate_all_refs(); // ensure all refs exist
        stream.instantiate_if_null();
    }

    // main execution function, ensure device ready and run
    void compute() {
        // ready_device();      // ensure device ready
        derived()._compute();
    }

#pragma region REFLECTION_REGISTRY // properties and methods reflected at the end the class

    // return properties
    // note we must used "Derived" for properties

#define PROPERTY(NAME) \
    Property<Derived, &Derived::NAME> { #NAME, &Derived::NAME }

    static constexpr auto properties() {
        return std::tuple_cat(Derived::_properties(), // CRTP requirement
                              std::tuple{
                                  // ================================================================
                                  // [Default Properties]
                                  // ----------------------------------------------------------------
                                  PROPERTY(stream),
                                  // ================================================================
                              });
    }

#undef PROPERTY

    // return methods
    // ⚠️ note we MUST reference GNC_Base here for methods (not Derived)
#define METHOD(FUNC) \
    Method<GNC_Base, &GNC_Base::FUNC> { #FUNC }

    static constexpr auto methods() {
        return std::tuple_cat(Derived::_methods(), // CRTP requirement
                              std::tuple{
                                  // ================================================================
                                  // [Default Methods]
                                  // ----------------------------------------------------------------
                                  //   Method<GNC_Base, &GNC_Base::_instance_test_1>{"_instance_test_1"},
                                  METHOD(_instance_test_1),
                                  METHOD(_instance_test_2),
                                  METHOD(_return_int_test),
                                  METHOD(instantiate_all_refs),
                                  METHOD(_copy_pars_and_array_ptrs_to_structs),
                                  // ================================================================
                              });
    }

#undef METHOD

#pragma endregion
};

} // namespace gnc
